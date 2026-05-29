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


def _dataset_names(c, dataset):
    """Return the list of dataset names to process."""
    if dataset:
        return [dataset]
    return list(c.config.get("datasets", {}).keys())


def _dataset_config(c, dataset_name):
    """Return (model_spec, subjects_str, task_names) for a dataset.

    task_names comes from the model's Input.task list; invoke.yaml may override
    it with a task_names key (comma-separated) to restrict or reorder tasks.
    """
    from analysis.model_spec import ModelSpec

    ds = c.config.get("datasets", {}).get(dataset_name, {})
    model_path = ds.get("model")
    if not model_path:
        raise ValueError(f"No model path configured for dataset '{dataset_name}'")
    model_spec = ModelSpec(model_path)
    subjects_str = ds.get("subjects", "")
    task_names_str = ds.get("task_names", "")
    task_names = [t.strip() for t in task_names_str.split(",")] if task_names_str else model_spec.tasks()
    return model_spec, subjects_str, task_names


def _cneuromod_paths(c, dataset_name):
    source_dir = Path(c.config.get("source_data_dir"))
    repo = source_dir / "cneuromod.all"
    bids_dir = repo / dataset_name / "bids"
    fmriprep_dir = repo / dataset_name / "fmriprep"
    return repo, bids_dir, fmriprep_dir


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

@task
def fetch(c, subjects=None, tasks=None, smoke=False, dataset=None):
    """Download Fedorenko atlas and get CNeuroMod fMRI data via DataLad."""
    from airoh.utils import download_data

    # ---- Fedorenko atlas (small, always download) ----
    download_data(c, "fedorenko_atlas_nii")
    download_data(c, "fedorenko_atlas_txt")

    # ---- DataLad superdataset ----
    source_dir = Path(c.config.get("source_data_dir"))
    repo = source_dir / "cneuromod.all"

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

    ssh_url = c.config.get("ria_sequoia_ssh_url", "")

    for ds_name in _dataset_names(c, dataset):
        _, bids_dir, fmriprep_dir = _cneuromod_paths(c, ds_name)
        model_spec, subjects_str, default_tasks = _dataset_config(c, ds_name)

        # Install subdatasets (metadata only, no data yet)
        for subds in [f"{ds_name}/bids", f"{ds_name}/fmriprep"]:
            c.run(f"datalad -C {repo} get -n {subds}", warn=True)

        # Configure RIA SSH remotes (idempotent)
        if ssh_url:
            for subds_path in [bids_dir, fmriprep_dir]:
                _configure_ria_ssh(c, subds_path, "ria-sequoia-storage", ssh_url)

        all_subjects = _parse_list(subjects, subjects_str)
        all_tasks = [t.strip() for t in tasks.split(",")] if tasks else default_tasks

        if smoke:
            all_subjects = all_subjects[:1]

        patterns = model_spec.file_patterns()
        bold_pat = patterns.get("bold", "*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
        mask_pat = patterns.get("mask", "*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
        conf_pat = patterns.get("confounds", "*_desc-confounds_timeseries.tsv")

        for subj in all_subjects:
            for task_name in all_tasks:
                func = f"{subj}/ses-*/func"

                events = sorted(bids_dir.glob(f"{func}/*task-{task_name}*_events.tsv"))
                fmriprep_files = sorted(
                    list(fmriprep_dir.glob(f"{func}/*task-{task_name}{bold_pat}"))
                    + list(fmriprep_dir.glob(f"{func}/*task-{task_name}{mask_pat}"))
                    + list(fmriprep_dir.glob(f"{func}/*task-{task_name}{conf_pat}"))
                )

                if not events and not fmriprep_files:
                    print(f"  [SKIP] No files found for {subj}/{task_name}")
                    continue

                if smoke:
                    # Restrict to the first session that actually has data for this task
                    first_ses = next(
                        (p for p in (events or fmriprep_files)[0].parts if p.startswith("ses-")),
                        None,
                    )
                    if first_ses:
                        events = [f for f in events if first_ses in f.parts]
                        fmriprep_files = [f for f in fmriprep_files if first_ses in f.parts]
                    ses_label = first_ses or "ses-*"
                else:
                    ses_label = "ses-*"

                print(f"  [{ds_name}] git annex get: {subj} / {ses_label} / {task_name}")

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
def run_glm(c, subjects=None, tasks=None, smoke=False, dataset=None):
    """Fit session-level first-level GLM for each subject; skip if outputs exist."""
    from analysis.glm import find_sessions_for_subject, run_session_glm
    import nibabel as nib

    output_dir = Path(c.config.get("output_data_dir"))
    tr = float(c.config.get("tr"))
    smoothing_fwhm = float(c.config.get("smoothing_fwhm"))
    n_compcor = int(c.config.get("n_compcor"))

    for ds_name in _dataset_names(c, dataset):
        _, bids_dir, fmriprep_dir = _cneuromod_paths(c, ds_name)
        model_spec, subjects_str, default_tasks = _dataset_config(c, ds_name)
        ds_output = output_dir / ds_name

        all_subjects = _parse_list(subjects, subjects_str)
        all_tasks = [t.strip() for t in tasks.split(",")] if tasks else default_tasks

        if smoke:
            all_subjects = all_subjects[:1]

        for subj in all_subjects:
            for task_name in all_tasks:
                sessions = find_sessions_for_subject(
                    subj, task_name, fmriprep_dir, bids_dir, model_spec
                )
                if not sessions:
                    print(f"  [SKIP] No sessions found for {subj}/{task_name}")
                    continue

                ses_list = sorted(sessions.items())
                if smoke:
                    ses_list = ses_list[:1]

                for ses_id, ses_data in ses_list:
                    session_dir = ds_output / subj / ses_id / task_name
                    first_contrast = model_spec.subject_contrasts(task_name)[0]
                    sentinel = (
                        session_dir /
                        f"{subj}_{ses_id}_task-{task_name}_contrast-{first_contrast}_stat-z.nii.gz"
                    )
                    if sentinel.exists():
                        print(f"  [SKIP] {subj}/{ses_id}/{task_name} (outputs exist)")
                        continue

                    session_dir.mkdir(parents=True, exist_ok=True)
                    runs = ses_data["runs"]
                    if smoke:
                        runs = runs[:1]

                    bold_files = [str(r["bold"]) for r in runs]
                    events_files = [str(r["events"]) for r in runs]
                    mask_file = str(ses_data["mask"]) if ses_data["mask"] else None

                    try:
                        results = run_session_glm(
                            subj, ses_id, task_name,
                            bold_files=bold_files,
                            events_files=events_files,
                            mask_file=mask_file,
                            model_spec=model_spec,
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
def run_subject(c, subjects=None, tasks=None, smoke=False, dataset=None):
    """Average session z-maps into subject-level fixed-effects maps; skip if outputs exist."""
    from analysis.glm import run_subject_level_fixed_effects

    output_dir = Path(c.config.get("output_data_dir"))

    for ds_name in _dataset_names(c, dataset):
        model_spec, subjects_str, default_tasks = _dataset_config(c, ds_name)
        ds_output = output_dir / ds_name

        all_subjects = _parse_list(subjects, subjects_str)
        all_tasks = [t.strip() for t in tasks.split(",")] if tasks else default_tasks

        if smoke:
            all_subjects = all_subjects[:1]

        for subj in all_subjects:
            for task_name in all_tasks:
                for contrast in model_spec.subject_contrasts(task_name):
                    out_path = (
                        ds_output / subj / "subject_level" / task_name /
                        f"{subj}_task-{task_name}_contrast-{contrast}_stat-z.nii.gz"
                    )
                    if out_path.exists():
                        print(f"  [SKIP] {subj}/{task_name}/{contrast} (output exists)")
                        continue
                    run_subject_level_fixed_effects(
                        subj, task_name, contrast, ds_output, ds_output
                    )


# ---------------------------------------------------------------------------
# run-froi  (Fedorenko fROI extraction, chunk = subject)
# ---------------------------------------------------------------------------

@task
def run_froi(c, subjects=None, tasks=None, smoke=False, dataset=None):
    """Extract Fedorenko language fROI masks from subject-level z-maps; skip if outputs exist."""
    from analysis.froi import load_fed_labels, extract_subject_frois, extract_session_frois

    source_dir = Path(c.config.get("source_data_dir"))
    output_dir = Path(c.config.get("output_data_dir"))
    top_percent = float(c.config.get("froi_top_percent"))

    atlas_nii = source_dir / "fedorenko" / "allParcels-language-SN220.nii"
    atlas_txt = source_dir / "fedorenko" / "allParcels-language-SN220.txt"

    if not atlas_nii.exists() or not atlas_txt.exists():
        print("[ERROR] Fedorenko atlas not found — run `invoke fetch` first")
        return

    fed_labels = load_fed_labels(atlas_txt)

    for ds_name in _dataset_names(c, dataset):
        model_spec, subjects_str, default_tasks = _dataset_config(c, ds_name)
        ds_output = output_dir / ds_name
        froi_dir = ds_output / "fedorenko_frois"
        froi_dir.mkdir(parents=True, exist_ok=True)

        all_subjects = _parse_list(subjects, subjects_str)
        all_tasks = [t.strip() for t in tasks.split(",")] if tasks else default_tasks

        if smoke:
            all_subjects = all_subjects[:1]

        for subj in all_subjects:
            for task_name in all_tasks:
                for contrast in model_spec.froi_contrasts(task_name):

                    # --- Session-level fROIs ---
                    ses_dirs = sorted((ds_output / subj).glob("ses-*"))
                    if smoke:
                        ses_dirs = ses_dirs[:1]

                    for ses_dir in ses_dirs:
                        ses_id = ses_dir.name
                        if list(froi_dir.glob(
                            f"{subj}_{ses_id}_task-{task_name}_contrast-{contrast}_parcel-*_top.nii.gz"
                        )):
                            print(f"  [SKIP] Session fROIs for {subj}/{ses_id}/{task_name}/{contrast} (outputs exist)")
                            continue
                        extract_session_frois(
                            subj, ses_id, task_name, contrast,
                            glm_dir=ds_output,
                            out_dir=froi_dir,
                            atlas_nii=atlas_nii,
                            fed_labels=fed_labels,
                            top_percent=top_percent,
                        )

                    # --- Subject-level fROIs ---
                    if list(froi_dir.glob(
                        f"{subj}_task-{task_name}_contrast-{contrast}_parcel-*_top_mean.nii.gz"
                    )):
                        print(f"  [SKIP] Subject fROIs for {subj}/{task_name}/{contrast} (outputs exist)")
                        continue
                    extract_subject_frois(
                        subj, task_name, contrast,
                        glm_dir=ds_output,
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
# run-figures
# ---------------------------------------------------------------------------
@task
def run_figures(c, figures=None):
    """Generate publication figures into output_data/figures_manuscript/."""
    from airoh.utils import run_notebooks as airoh_run_notebooks, ensure_dir_exist

    repo_root     = Path.cwd()
    notebooks_dir = repo_root / c.config.get("notebooks_dir")
    output_dir    = (repo_root / c.config.get("output_data_dir")).resolve()
    source_dir    = (repo_root / c.config.get("source_data_dir")).resolve()
    figures_dir   = (repo_root / c.config.get("figures_output_dir",
                     "output_data/figures_manuscript")).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Patch config with absolute paths so airoh injects them correctly
    c.config["output_data_dir"]    = str(output_dir)
    c.config["source_data_dir"]    = str(source_dir)
    c.config["figures_output_dir"] = str(figures_dir)

    nat_dir = c.config.get("naturalistic_data_dir", "")
    if nat_dir:
        c.config["naturalistic_data_dir"] = str(Path(nat_dir).resolve())

    keys = ["output_data_dir", "source_data_dir",
            "figures_output_dir", "naturalistic_data_dir"]

    if figures:
        requested = [f.strip() for f in figures.split(",")]
        for fig_name in requested:
            nb = notebooks_dir / f"{fig_name}.ipynb"
            if not nb.exists():
                print(f"  [SKIP] Notebook not found: {nb}")
                continue
            out_dir = figures_dir / fig_name
            if out_dir.exists() and any(out_dir.iterdir()):
                print(f"  [SKIP] {fig_name} (outputs exist)")
                continue
            env_str = " ".join(
                f'{k.upper()}="{c.config.get(k, "")}"' for k in keys
            )
            c.run(
                f'env {env_str} uv run jupyter nbconvert '
                f'--to notebook --execute --inplace {nb}',
                warn=True,
            )
    else:
        airoh_run_notebooks(c, notebooks_dir, figures_dir, keys=keys)


@task
def clean_figures(c):
    """Remove output_data/figures_manuscript/."""
    from airoh.utils import clean_folder
    clean_folder(c, "figures_output_dir", "")

    
# ---------------------------------------------------------------------------
# run  /  run-smoke
# ---------------------------------------------------------------------------

@task
def run(c, smoke=False, dataset=None):
    """Full pipeline: fetch → run-glm → run-subject → run-froi → run-notebooks."""
    fetch(c, smoke=smoke, dataset=dataset)
    run_glm(c, smoke=smoke, dataset=dataset)
    run_subject(c, smoke=smoke, dataset=dataset)
    run_froi(c, smoke=smoke, dataset=dataset)
    run_notebooks(c)
    run_figures(c)
    print("Pipeline complete.")
    

@task
def run_smoke(c, dataset=None):
    """Smoke test: minimal end-to-end pass (first subject only).

    Pass --dataset to restrict to one dataset, e.g.:
      invoke run-smoke --dataset hcptrt
    """
    run(c, smoke=True, dataset=dataset)


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

@task
def clean_glm(c):
    """Remove session-level GLM z-maps from output_data/."""
    from airoh.utils import clean_folder
    for ds_name in _dataset_names(c, None):
        clean_folder(c, "output_data_dir", f"{ds_name}/sub-*/ses-*/*/*.nii.gz")


@task
def clean_subject(c):
    """Remove subject-level fixed-effects maps from output_data/."""
    from airoh.utils import clean_folder
    for ds_name in _dataset_names(c, None):
        clean_folder(c, "output_data_dir", f"{ds_name}/sub-*/subject_level/*/*.nii.gz")


@task
def clean_froi(c):
    """Remove Fedorenko fROI masks from output_data/."""
    from airoh.utils import clean_folder
    for ds_name in _dataset_names(c, None):
        clean_folder(c, "output_data_dir", f"{ds_name}/fedorenko_frois/*.nii.gz")


@task(pre=[clean_glm, clean_subject, clean_froi, clean_figures])
def clean(c):
    """Remove all computed outputs."""
    pass

@task
def clean_source(c):
    """Remove downloaded source data (Fedorenko atlas + DataLad files)."""
    from airoh.utils import clean_folder
    clean_folder(c, "source_data_dir", "fedorenko/")
    source_dir = Path(c.config.get("source_data_dir"))
    repo = source_dir / "cneuromod.all"
    if repo.exists():
        print(f"[NOTE] DataLad repo at {repo} was not removed — delete manually if needed.")
