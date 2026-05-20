import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn.image import resample_to_img


def load_fed_labels(txt_path):
    """Load Fedorenko parcel names from text file; returns {1-based-index: name}."""
    with open(txt_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return {i: name for i, name in enumerate(lines, start=1)}


def _load_fed_atlas(atlas_nii, reference_img):
    """Load and resample Fedorenko atlas to match reference image space."""
    atlas_img = nib.load(str(atlas_nii))
    if (atlas_img.shape != reference_img.shape or
            not np.allclose(atlas_img.affine, reference_img.affine)):
        atlas_img = resample_to_img(atlas_img, reference_img, interpolation="nearest")
    return atlas_img.get_fdata(), atlas_img


def _make_top_voxel_mask(contrast_data, parcel_mask, top_percent):
    """Select top `top_percent` fraction of positive voxels within a parcel."""
    vals = contrast_data[parcel_mask]
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.zeros_like(contrast_data, dtype=bool)
    k = max(1, int(np.floor(top_percent * vals.size)))
    thresh = np.sort(vals)[-k]
    return parcel_mask & (contrast_data >= thresh)


def extract_session_frois(subject, session, task, contrast,
                          glm_dir, out_dir, atlas_nii, fed_labels, top_percent=0.10):
    """Extract per-parcel fROI masks from one session's z-map."""
    glm_dir = Path(glm_dir)
    out_dir = Path(out_dir)
    z_map_path = (
        glm_dir / subject / session / task /
        f"{subject}_{session}_task-{task}_contrast-{contrast}_stat-z.nii.gz"
    )
    if not z_map_path.exists():
        print(f"  [SKIP] z-map not found: {z_map_path.name}")
        return

    img = nib.load(str(z_map_path))
    data = img.get_fdata()
    atlas_data, _ = _load_fed_atlas(atlas_nii, img)

    for label, roi_name in fed_labels.items():
        parcel_mask = (atlas_data == float(label)).astype(bool)
        if parcel_mask.sum() == 0:
            continue
        top_mask = _make_top_voxel_mask(data, parcel_mask, top_percent)
        if top_mask.sum() == 0:
            continue
        fname = (
            f"{subject}_{session}_task-{task}_contrast-{contrast}"
            f"_parcel-{roi_name}_top.nii.gz"
        )
        mask_img = nib.Nifti1Image(top_mask.astype(np.uint8), img.affine, img.header)
        nib.save(mask_img, str(out_dir / fname))


def extract_subject_frois(subject, task, contrast,
                          glm_dir, out_dir, atlas_nii, fed_labels, top_percent=0.10):
    """Extract per-parcel fROI masks using subject-level (cross-session) z-map."""
    glm_dir = Path(glm_dir)
    out_dir = Path(out_dir)

    subj_level_map = (
        glm_dir / subject / "subject_level" / task /
        f"{subject}_task-{task}_contrast-{contrast}_stat-z.nii.gz"
    )

    if subj_level_map.exists():
        img = nib.load(str(subj_level_map))
        data = img.get_fdata()
    else:
        # Fall back to averaging session-level maps
        vols, ref_img = [], None
        for ses_dir in sorted((glm_dir / subject).glob("ses-*")):
            z_map = (
                ses_dir / task /
                f"{subject}_{ses_dir.name}_task-{task}_contrast-{contrast}_stat-z.nii.gz"
            )
            if z_map.exists():
                img = nib.load(str(z_map))
                vols.append(img.get_fdata())
                ref_img = img
        if not vols:
            print(f"  [SKIP] No z-maps for {subject}/{task}/{contrast}")
            return
        data = np.mean(np.stack(vols, axis=0), axis=0)
        img = ref_img

    atlas_data, _ = _load_fed_atlas(atlas_nii, img)

    for label, roi_name in fed_labels.items():
        parcel_mask = (atlas_data == float(label)).astype(bool)
        if parcel_mask.sum() == 0:
            continue
        top_mask = _make_top_voxel_mask(data, parcel_mask, top_percent)
        if top_mask.sum() == 0:
            continue
        fname = (
            f"{subject}_task-{task}_contrast-{contrast}"
            f"_parcel-{roi_name}_top_mean.nii.gz"
        )
        mask_img = nib.Nifti1Image(top_mask.astype(np.uint8), img.affine, img.header)
        nib.save(mask_img, str(out_dir / fname))
