import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.fmriprep import load_confounds


def load_confounds_extended(bold_file, n_compcor=6):
    confounds, _ = load_confounds(
        bold_file,
        strategy=["motion", "high_pass", "compcor"],
        motion="basic",
        compcor="anat_combined",
        n_compcor=n_compcor,
    )
    return confounds


def prepare_events(event_file, transformations):
    """Load events TSV and apply Replace/Drop instructions from a BIDS SM node."""
    df = pd.read_csv(event_file, sep="\t")
    bids = df[["onset", "duration", "trial_type"]].copy()
    bids["onset"] = pd.to_numeric(bids["onset"], errors="coerce")
    bids["duration"] = pd.to_numeric(bids["duration"], errors="coerce")
    bids = bids.dropna(subset=["onset", "duration"])
    bids.loc[bids["duration"] <= 0, "duration"] = 1.0

    for instr in transformations:
        name = instr.get("Name")
        if name == "Replace" and "trial_type" in instr.get("Input", []):
            mapping = instr.get("Map", {})
            bids["trial_type"] = bids["trial_type"].replace(mapping)
        elif name == "Drop" and "trial_type" in instr.get("Input", []):
            values = instr.get("Values", [])
            bids = bids[~bids["trial_type"].isin(values)]

    return bids.reset_index(drop=True)


def build_contrasts(run_contrasts, design_matrices):
    """Build nilearn contrast expressions from a BIDS SM Contrasts list.

    ConditionList entries use the 'trial_type.value' convention; this strips
    the prefix to get the raw column name as it appears in the design matrix.
    """
    all_cols = set()
    for dm in design_matrices:
        all_cols.update(dm.columns)

    contrasts = {}
    for spec in run_contrasts:
        cond_list = [c.split(".", 1)[-1] for c in spec["ConditionList"]]
        weights = spec["Weights"]
        if not all(c in all_cols for c in cond_list):
            continue
        if len(cond_list) == 1:
            contrasts[spec["Name"]] = cond_list[0]
        else:
            terms = [f"{w:+g}*{c}" if w != 1 else c for w, c in zip(weights, cond_list)]
            # Build expression: w1*A + w2*B → "A - B" or "A + B" etc.
            expr_parts = []
            for w, c in zip(weights, cond_list):
                if not expr_parts:
                    expr_parts.append(c if w == 1 else f"-{c}" if w == -1 else f"{w}*{c}")
                else:
                    if w == 1:
                        expr_parts.append(f"+ {c}")
                    elif w == -1:
                        expr_parts.append(f"- {c}")
                    else:
                        expr_parts.append(f"+ {w}*{c}" if w > 0 else f"- {abs(w)}*{c}")
            contrasts[spec["Name"]] = " ".join(expr_parts)
    return contrasts


def run_session_glm(subject, session, task, bold_files, events_files, mask_file,
                    model_spec, tr=1.49, smoothing_fwhm=5.0, n_compcor=6):
    """Fit a first-level GLM for one subject/session/task.

    bold_files and events_files are lists (one entry per run).
    """
    print(f"  GLM: {subject}/{session}/{task} ({len(bold_files)} run(s))")
    transformations = model_spec.transformations(task)
    events = [prepare_events(ef, transformations) for ef in events_files]
    confounds = [load_confounds_extended(bf, n_compcor=n_compcor) for bf in bold_files]

    fmri_glm = FirstLevelModel(
        t_r=tr,
        mask_img=mask_file,
        drift_model=None,
        smoothing_fwhm=smoothing_fwhm,
    )
    fmri_glm.fit(bold_files, events=events, confounds=confounds)

    run_contrasts = model_spec.run_contrasts(task)
    contrast_map = build_contrasts(run_contrasts, fmri_glm.design_matrices_)
    results = {}
    for name, expr in contrast_map.items():
        try:
            results[name] = fmri_glm.compute_contrast(expr, output_type="z_score")
        except Exception as e:
            print(f"    [ERROR] contrast {name}: {e}")
    return results


def find_sessions_for_subject(subject, task, fmriprep_dir, bids_dir, model_spec):
    """Return session data for all discoverable sessions of subject/task.

    Returns:
        {ses_id: {"runs": [{"bold": path, "events": path, "run_id": str}], "mask": path}}

    For datasets without a run dimension (langlocalizer) each session has one
    entry in "runs" and run_id is None.
    """
    patterns = model_spec.file_patterns()
    bold_suffix = patterns.get("bold", "*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    mask_suffix = patterns.get("mask", "*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")

    subj_fmriprep = Path(fmriprep_dir) / subject
    subj_bids = Path(bids_dir) / subject

    sessions = {}

    if model_spec.has_run_dimension():
        # hcptrt-style: multiple runs per session
        # Discover via events files (always in BIDS sourcedata)
        bold_end = bold_suffix.split("*")[-1]
        mask_end = mask_suffix.split("*")[-1]
        for events_file in sorted(subj_bids.rglob(f"{subject}_ses-*_task-{task}_run-*_events.tsv")):
            m = re.search(r"(ses-[^_]+).*?(run-[^_]+)", events_file.name)
            if not m:
                continue
            ses_id, run_id = m.group(1), m.group(2)
            func_dir = subj_fmriprep / ses_id / "func"
            bold_files = list(func_dir.glob(f"*_task-{task}_{run_id}*{bold_end}"))
            mask_files = list(func_dir.glob(f"*_task-{task}_{run_id}*{mask_end}"))
            if not bold_files:
                # bids events may use run-01 while fmriprep uses run-1 — try stripped version
                run_id_short = re.sub(r"run-0*(\d+)", r"run-\1", run_id)
                if run_id_short != run_id:
                    bold_files = list(func_dir.glob(f"*_task-{task}_{run_id_short}*{bold_end}"))
                    mask_files = list(func_dir.glob(f"*_task-{task}_{run_id_short}*{mask_end}"))
                    if bold_files:
                        run_id = run_id_short
            if not bold_files:
                print(f"  [WARN] No bold for {subject}/{ses_id}/{task}/{run_id} — skipping run")
                continue
            if ses_id not in sessions:
                sessions[ses_id] = {"runs": [], "mask": mask_files[0] if mask_files else None}
            sessions[ses_id]["runs"].append({
                "bold": bold_files[0],
                "events": events_file,
                "run_id": run_id,
            })
    else:
        # langlocalizer-style: one run per session (no run-* in filenames)
        for bold_file in sorted(subj_fmriprep.rglob(f"{subject}_ses-*_task-{task}{bold_suffix}")):
            m = re.search(r"(ses-[^_]+)", bold_file.name)
            if not m:
                continue
            ses_id = m.group(1)
            events_files = list(subj_bids.rglob(f"{subject}_{ses_id}_task-{task}_events.tsv"))
            if not events_files:
                print(f"  [WARN] No events for {subject}/{ses_id}/{task} — skipping")
                continue
            mask_files = list(subj_fmriprep.rglob(f"{subject}_{ses_id}_task-{task}{mask_suffix}"))
            sessions[ses_id] = {
                "runs": [{"bold": bold_file, "events": events_files[0], "run_id": None}],
                "mask": mask_files[0] if mask_files else None,
            }

    return sessions


def run_subject_level_fixed_effects(subject, task, contrast, glm_dir, output_dir):
    """Average session-level z-maps into a subject-level fixed-effects map."""
    glm_dir = Path(glm_dir)
    output_dir = Path(output_dir)
    z_map_paths = sorted(
        glm_dir.glob(
            f"{subject}/ses-*/{task}/"
            f"{subject}_ses-*_task-{task}_contrast-{contrast}_stat-z.nii.gz"
        )
    )
    if len(z_map_paths) < 2:
        print(f"  [SKIP] {subject}/{task}/{contrast}: need ≥2 sessions (found {len(z_map_paths)})")
        return None

    imgs = [nib.load(str(p)) for p in z_map_paths]
    mean_data = np.mean(np.stack([img.get_fdata() for img in imgs], axis=0), axis=0)
    out_img = nib.Nifti1Image(mean_data, imgs[0].affine, imgs[0].header)

    out_dir = output_dir / subject / "subject_level" / task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}_task-{task}_contrast-{contrast}_stat-z.nii.gz"
    nib.save(out_img, str(out_path))
    return out_path
