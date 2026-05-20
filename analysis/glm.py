import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.fmriprep import load_confounds


FUNC_PATTERN = "*_part-mag_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
MASK_PATTERN = "*_part-mag_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"

EVENTS_OF_INTEREST = {
    "listening": ["int", "degr"],
    "aliceFr": ["int", "degr", "tamil"],
    "aliceEn": ["int", "degr", "tamil"],
    "reading": ["word", "nonword"],
}

TASK_CONTRASTS = {
    "listening": {
        "singles": ["int", "degr"],
        "pairs": [
            ("int", "degr", "int-degr"),
            ("degr", "int", "degr-int"),
        ],
    },
    "aliceFr": {
        "singles": ["int", "degr", "tamil"],
        "pairs": [
            ("int", "degr", "int-degr"),
            ("degr", "int", "degr-int"),
            ("tamil", "degr", "tamil-degr"),
            ("tamil", "int", "tamil-int"),
        ],
    },
    "aliceEn": {
        "singles": ["int", "degr"],
        "pairs": [
            ("int", "degr", "int-degr"),
            ("degr", "int", "degr-int"),
            ("tamil", "degr", "tamil-degr"),
            ("tamil", "int", "tamil-int"),
        ],
    },
    "reading": {
        "singles": ["word", "nonword"],
        "pairs": [
            ("word", "nonword", "word-nonword"),
            ("nonword", "word", "nonword-word"),
        ],
    },
}

CONTRASTS_TO_AVERAGE = {
    "listening": ["int-degr", "degr-int", "int", "degr"],
    "aliceFr":   ["int-degr", "degr-int", "int", "degr"],
    "aliceEn":   ["int-degr", "degr-int", "int", "degr"],
    "reading":   ["word-nonword", "nonword-word", "word", "nonword"],
}


def load_confounds_extended(bold_file, n_compcor=6):
    confounds, _ = load_confounds(
        bold_file,
        strategy=["motion", "high_pass", "compcor"],
        motion="basic",
        compcor="anat_combined",
        n_compcor=n_compcor,
    )
    return confounds


def _convert_listening_alice_events(event_file):
    df = pd.read_csv(event_file, sep="\t")
    bids = df[["onset", "duration", "trial_type"]].copy()
    bids["onset"] = pd.to_numeric(bids["onset"], errors="coerce")
    bids["duration"] = pd.to_numeric(bids["duration"], errors="coerce")
    bids = bids.dropna()
    # Some events in the source data have invalid negative durations
    bids.loc[bids["duration"] <= 0, "duration"] = 1.0
    return bids


def _convert_reading_events(event_file):
    df = pd.read_csv(event_file, sep="\t")
    bids = df[["onset", "duration", "trial_type"]].copy()
    bids["onset"] = pd.to_numeric(bids["onset"], errors="coerce")
    bids["duration"] = pd.to_numeric(bids["duration"], errors="coerce")
    bids = bids.dropna()
    bids.loc[bids["duration"] <= 0, "duration"] = 1.0
    # 'non-word' → 'nonword': hyphens are invalid in nilearn contrast expressions
    bids["trial_type"] = bids["trial_type"].str.replace("-", "", regex=False)
    return bids


def prepare_events(event_file, task):
    if task in ["listening", "aliceFr", "aliceEn"]:
        return _convert_listening_alice_events(event_file)
    elif task == "reading":
        return _convert_reading_events(event_file)
    raise ValueError(f"Unknown task: {task}")


def build_contrasts_for_task(task, design_matrices):
    if task not in TASK_CONTRASTS:
        return {}
    spec = TASK_CONTRASTS[task]
    all_cols = set()
    for dm in design_matrices:
        all_cols.update(dm.columns)
    contrasts = {}
    for cond in spec.get("singles", []):
        if cond in all_cols:
            contrasts[cond] = cond
    for cond1, cond2, name in spec.get("pairs", []):
        if cond1 in all_cols and cond2 in all_cols:
            contrasts[name] = f"{cond1} - {cond2}"
    return contrasts


def run_session_glm(subject, session, task, bold_file, events_file, mask_file,
                    tr=1.49, smoothing_fwhm=5.0, n_compcor=6):
    print(f"  GLM: {subject}/{session}/{task}")
    events = prepare_events(events_file, task)
    confounds = load_confounds_extended(bold_file, n_compcor=n_compcor)

    fmri_glm = FirstLevelModel(
        t_r=tr,
        mask_img=mask_file,
        drift_model=None,
        smoothing_fwhm=smoothing_fwhm,
    )
    fmri_glm.fit(bold_file, events=events, confounds=confounds)

    contrasts = build_contrasts_for_task(task, fmri_glm.design_matrices_)
    results = {}
    for name, expr in contrasts.items():
        try:
            results[name] = fmri_glm.compute_contrast(expr, output_type="z_score")
        except Exception as e:
            print(f"    [ERROR] contrast {name}: {e}")
    return results


def find_sessions_for_subject(subject, task, fmriprep_dir, bids_dir):
    """Return {session_id: {bold, events, mask}} for all discoverable sessions."""
    subject_root = Path(fmriprep_dir) / subject
    source_root = Path(bids_dir) / subject

    bold_files = sorted(subject_root.rglob(f"{subject}_ses-*_task-{task}{FUNC_PATTERN}"))
    sessions = {}
    for bold_file in bold_files:
        ses_id = next(p for p in bold_file.name.split("_") if p.startswith("ses-"))
        events_files = list(source_root.rglob(f"{subject}_{ses_id}_task-{task}_events.tsv"))
        if not events_files:
            print(f"  [WARN] No events for {subject}/{ses_id}/{task} — skipping")
            continue
        mask_files = list(subject_root.rglob(f"{subject}_{ses_id}_task-{task}{MASK_PATTERN}"))
        sessions[ses_id] = {
            "bold": bold_file,
            "events": events_files[0],
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
