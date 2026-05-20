# Output Data

## GLM outputs

- `sub-{sub}/ses-{ses}/{task}/sub-{sub}_ses-{ses}_task-{task}_contrast-{contrast}_stat-z.nii.gz`
  Session-level z-score maps for each contrast, produced by `invoke run-glm`.

- `sub-{sub}/subject_level/{task}/sub-{sub}_task-{task}_contrast-{contrast}_stat-z.nii.gz`
  Subject-level fixed-effects maps (average across sessions), produced by `invoke run-subject`.

## fROI outputs

- `fedorenko_frois/sub-{sub}_task-{task}_contrast-{contrast}_parcel-{roi}_top_mean.nii.gz`
  Top-voxel fROI masks (subject-level, using Fedorenko language parcels), produced by `invoke run-froi`.

## Figures

- `figures_glm/sub-{sub}_subject_level_summary.png`
  Multi-task summary plots of subject-level contrast maps, produced by `invoke run-notebooks`.

📝 All NIfTI and PNG files are **ignored by Git** (see `.gitignore`).
