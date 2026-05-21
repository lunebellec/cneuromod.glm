from pathlib import Path
from invoke import task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_list(value, default_str):
    """Parse a comma-separated string or return the config default."""
    if value:
        return [v.strip() for v in value.split(",")]
    return [v.strip() for v in default_str.split(",")]


def _configure_ria_ssh(c, subds_path, remote_name, ssh_url):
    """Reconfigure an ORA remote to use SSH if the current URL differs."""
    import subprocess, re

    # Parse host and path from ria+ssh://host/path
    m = re.match(r"ria\+ssh://([^/]+)(/.+)", ssh_url)
    if not m:
        print(f"  [ERROR] Cannot parse SSH URL: {ssh_url}")
        return
    host, path = m.group(1), m.group(2)

    # Verify passwordless SSH access and that the path is readable
    probe = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, f"ls {path}"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        print(
            f"\n[ERROR] Cannot reach {host}:{path} via SSH.\n"
            f"  Make sure you have passwordless SSH access to '{host}' and that\n"
            f"  the path '{path}' exists and is readable.\n"
            f"  Quick test: ssh {host} ls {path}\n"
            f"  SSH error: {probe.stderr.strip()}"
        )
        return

    # Check if already configured
    result = subprocess.run(
        ["git", "cat-file", "-p", "refs/heads/git-annex:remote.log"],
        cwd=str(subds_path), capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if f"name={remote_name}" in line and f"url={ssh_url}" in line:
            print(f"  [OK] {remote_name} already configured for SSH ({subds_path.name})")
            return

    print(f"  Configuring {remote_name} → {ssh_url} ({subds_path.name})")
    c.run(
        f"git -C {subds_path} annex enableremote {remote_name} url={ssh_url}",
        warn=True,
    )


def _cneuromod_paths(c):
    source_dir = Path(c.config.get("source_data_dir"))
    dataset = c.config.get("cneuromod_dataset")
    repo = source_dir / "cneuromod.all"
    bids_dir = repo / dataset / "bids"
    fmriprep_dir = repo / dataset / "fmriprep"
    return repo, bids_dir, fmriprep_dir


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

@task
def fetch(c, subjects=None, tasks=None, smoke=False):
    """Download Fedorenko atlas and get CNeuroMod fMRI data via DataLad."""
    from airoh.utils import download_data

    # ---- Fedorenko atlas (small, always download) ----
    download_data(c, "fedorenko_atlas_nii")
    download_data(c, "fedorenko_atlas_txt")

    # ---- DataLad superdataset ----
    repo, bids_dir, fmriprep_dir = _cneuromod_paths(c)

    if not repo.exists():
        print(f"Cloning cneuromod.all → {repo}  (SSH access to GitHub required)")
        c.run(
            f"datalad clone git@github.com:courtois-neuromod/cneuromod.all.git {repo}",
            warn=True,
        )
    else:
        print(f"DataLad repo already present: {repo}")

    if not repo.exists():
        print("[WARN] Could not clone cneuromod.all — skipping datalad get")
        return

    dataset = c.config.get("cneuromod_dataset")

    # Install subdatasets (metadata only, no data yet)
    for subds in [f"{dataset}/bids", f"{dataset}/fmriprep"]:
        c.run(f"datalad -C {repo} get -n {subds}", warn=True)

    # Configure RIA SSH remotes (idempotent)
    ssh_url = c.config.get("ria_sequoia_ssh_url", "")
    if ssh_url:
        for subds_path in [bids_dir, fmriprep_dir]:
            _configure_ria_ssh(c, subds_path, "ria-sequoia-storage", ssh_url)

    # Get files for each subject / task
    all_subjects = _parse_list(subjects, c.config.get("subjects"))
    all_tasks = _parse_list(tasks, c.config.get("task_names"))

    if smoke:
        all_subjects = all_subjects[:1]

    ses_pattern = "ses-001" if smoke else "ses-*"

    for subj in all_subjects:
        for task_name in all_tasks:
            print(f"  git annex get: {subj} / {ses_pattern} / {task_name}")
            func = f"{subj}/{ses_pattern}/func"

            events = list(bids_dir.glob(f"{func}/*task-{task_name}*_events.tsv"))
            fmriprep_files = (
                list(fmriprep_dir.glob(
                    f"{func}/*task-{task_name}*_part-mag_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                ))
                + list(fmriprep_dir.glob(
                    f"{func}/*task-{task_name}*_part-mag_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
                ))
                + list(fmriprep_dir.glob(
                    f"{func}/*task-{task_name}*_part-mag_desc-confounds_timeseries.tsv"
                ))
            )

            if not events and not fmriprep_files:
                print(f"  [SKIP] No files found for {subj}/{ses_pattern}/{task_name}")
                continue

            if events:
                rel = " ".join(str(f.relative_to(bids_dir)) for f in events)
                c.run(f"git -C {bids_dir} annex get {rel}", warn=True)

            if fmriprep_files:
                rel = " ".join(str(f.relative_to(fmriprep_dir)) for f in fmriprep_files)
                c.run(f"git -C {fmriprep_dir} annex get {rel}", warn=True)


# ---------------------------------------------------------------------------
# run-glm  (session-level first-level GLM, chunk = subject)
# ---------------------------------------------------------------------------

@task
def run_glm(c, subjects=None, tasks=None, smoke=False):
    """Fit session-level first-level GLM for each subject; skip if outputs exist."""
    from analysis.glm import (
        find_sessions_for_subject, run_session_glm, CONTRASTS_TO_AVERAGE
    )
    import nibabel as nib

    _, bids_dir, fmriprep_dir = _cneuromod_paths(c)
    output_dir = Path(c.config.get("output_data_dir"))
    tr = float(c.config.get("tr"))
    smoothing_fwhm = float(c.config.get("smoothing_fwhm"))
    n_compcor = int(c.config.get("n_compcor"))

    all_subjects = _parse_list(subjects, c.config.get("subjects"))
    all_tasks = _parse_list(tasks, c.config.get("task_names"))

    if smoke:
        all_subjects = all_subjects[:1]

    for subj in all_subjects:
        for task_name in all_tasks:
            sessions = find_sessions_for_subject(subj, task_name, fmriprep_dir, bids_dir)
            if not sessions:
                print(f"  [SKIP] No sessions found for {subj}/{task_name}")
                continue

            ses_list = sorted(sessions.items())
            if smoke:
                ses_list = ses_list[:1]

            for ses_id, files in ses_list:
                session_dir = output_dir / subj / ses_id / task_name
                sentinel = session_dir / f"{subj}_{ses_id}_task-{task_name}_contrast-int-degr_stat-z.nii.gz"
                if sentinel.exists():
                    print(f"  [SKIP] {subj}/{ses_id}/{task_name} (outputs exist)")
                    continue

                session_dir.mkdir(parents=True, exist_ok=True)
                try:
                    results = run_session_glm(
                        subj, ses_id, task_name,
                        bold_file=str(files["bold"]),
                        events_file=str(files["events"]),
                        mask_file=str(files["mask"]) if files["mask"] else None,
                        tr=tr,
                        smoothing_fwhm=smoothing_fwhm,
                        n_compcor=n_compcor,
                    )
                    for contrast_name, z_map in results.items():
                        out_path = (
                            session_dir /
                            f"{subj}_{ses_id}_task-{task_name}_contrast-{contrast_name}_stat-z.nii.gz"
                        )
                        nib.save(z_map, str(out_path))
                        print(f"    Saved: {out_path.name}")
                except Exception as e:
                    print(f"  [ERROR] {subj}/{ses_id}/{task_name}: {e}")
                    import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# run-subject  (fixed-effects subject-level GLM, chunk = subject)
# ---------------------------------------------------------------------------

@task
def run_subject(c, subjects=None, tasks=None, smoke=False):
    """Average session z-maps into subject-level fixed-effects maps; skip if outputs exist."""
    from analysis.glm import run_subject_level_fixed_effects, CONTRASTS_TO_AVERAGE

    output_dir = Path(c.config.get("output_data_dir"))
    all_subjects = _parse_list(subjects, c.config.get("subjects"))
    all_tasks = _parse_list(tasks, c.config.get("task_names"))

    if smoke:
        all_subjects = all_subjects[:1]

    for subj in all_subjects:
        for task_name in all_tasks:
            for contrast in CONTRASTS_TO_AVERAGE.get(task_name, []):
                out_path = (
                    output_dir / subj / "subject_level" / task_name /
                    f"{subj}_task-{task_name}_contrast-{contrast}_stat-z.nii.gz"
                )
                if out_path.exists():
                    print(f"  [SKIP] {subj}/{task_name}/{contrast} (output exists)")
                    continue
                run_subject_level_fixed_effects(subj, task_name, contrast, output_dir, output_dir)


# ---------------------------------------------------------------------------
# run-froi  (Fedorenko fROI extraction, chunk = subject)
# ---------------------------------------------------------------------------

@task
def run_froi(c, subjects=None, tasks=None, smoke=False):
    """Extract Fedorenko language fROI masks from subject-level z-maps; skip if outputs exist."""
    from analysis.froi import load_fed_labels, extract_subject_frois

    source_dir = Path(c.config.get("source_data_dir"))
    output_dir = Path(c.config.get("output_data_dir"))
    top_percent = float(c.config.get("froi_top_percent"))

    atlas_nii = source_dir / "fedorenko" / "allParcels-language-SN220.nii"
    atlas_txt = source_dir / "fedorenko" / "allParcels-language-SN220.txt"

    if not atlas_nii.exists() or not atlas_txt.exists():
        print("[ERROR] Fedorenko atlas not found — run `invoke fetch` first")
        return

    fed_labels = load_fed_labels(atlas_txt)
    froi_dir = output_dir / "fedorenko_frois"
    froi_dir.mkdir(parents=True, exist_ok=True)

    all_subjects = _parse_list(subjects, c.config.get("subjects"))
    all_tasks = _parse_list(tasks, c.config.get("task_names"))

    if smoke:
        all_subjects = all_subjects[:1]

    FROI_CONTRASTS = {
        "aliceFr":   ["int-degr"],
        "aliceEn":   ["int-degr"],
        "listening": ["int-degr"],
        "reading":   ["word-nonword"],
    }

    for subj in all_subjects:
        for task_name in all_tasks:
            for contrast in FROI_CONTRASTS.get(task_name, []):
                sentinel = froi_dir / f"{subj}_task-{task_name}_contrast-{contrast}_parcel-*_top_mean.nii.gz"
                if list(froi_dir.glob(f"{subj}_task-{task_name}_contrast-{contrast}_parcel-*_top_mean.nii.gz")):
                    print(f"  [SKIP] fROIs for {subj}/{task_name}/{contrast} (outputs exist)")
                    continue
                extract_subject_frois(
                    subj, task_name, contrast,
                    glm_dir=output_dir,
                    out_dir=froi_dir,
                    atlas_nii=atlas_nii,
                    fed_labels=fed_labels,
                    top_percent=top_percent,
                )


# ---------------------------------------------------------------------------
# run-notebooks
# ---------------------------------------------------------------------------

@task
def run_notebooks(c):
    """Execute visualization notebooks; skip notebooks whose output folder exists."""
    from airoh.utils import run_notebooks as airoh_run_notebooks, ensure_dir_exist

    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()
    ensure_dir_exist(c, "output_data_dir")
    airoh_run_notebooks(c, notebooks_dir, output_dir, keys=["source_data_dir", "output_data_dir"])


# ---------------------------------------------------------------------------
# run  /  run-smoke
# ---------------------------------------------------------------------------

@task(pre=[fetch, run_glm, run_subject, run_froi, run_notebooks])
def run(c):
    """Full pipeline: fetch → run-glm → run-subject → run-froi → run-notebooks."""
    print("Pipeline complete.")


@task
def run_smoke(c):
    """Smoke test: minimal end-to-end pass (first subject only)."""
    fetch(c, smoke=True)
    run_glm(c, smoke=True)
    run_subject(c, smoke=True)
    run_froi(c, smoke=True)
    run_notebooks(c)


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

@task
def clean_glm(c):
    """Remove session-level GLM z-maps from output_data/."""
    from airoh.utils import clean_folder
    clean_folder(c, "output_data_dir", "sub-*/ses-*/*/*.nii.gz")


@task
def clean_subject(c):
    """Remove subject-level fixed-effects maps from output_data/."""
    from airoh.utils import clean_folder
    clean_folder(c, "output_data_dir", "sub-*/subject_level/*/*.nii.gz")


@task
def clean_froi(c):
    """Remove Fedorenko fROI masks from output_data/."""
    from airoh.utils import clean_folder
    clean_folder(c, "output_data_dir", "fedorenko_frois/*.nii.gz")


@task(pre=[clean_glm, clean_subject, clean_froi])
def clean(c):
    """Remove all computed outputs."""
    pass


@task
def clean_source(c):
    """Remove downloaded source data (Fedorenko atlas + DataLad files)."""
    from airoh.utils import clean_folder
    clean_folder(c, "source_data_dir", "fedorenko/")
    repo, _, _ = _cneuromod_paths(c)
    if repo.exists():
        print(f"[NOTE] DataLad repo at {repo} was not removed — delete manually if needed.")
