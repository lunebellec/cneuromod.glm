# Source Data

## `fedorenko/`

- `allParcels-language-SN220.nii` — Fedorenko language parcellation atlas (NIfTI).
  Downloaded from `https://www.evlab.mit.edu/s/allParcels-language-SN220-hgwm.nii`.
  Reference: Fedorenko et al. (2010), *Journal of Neurophysiology* 104(2): 1177–1194.
  **Note:** No explicit license is provided by the authors. A properly licensed deposit
  on Zenodo is planned for future reproducibility.
- `allParcels-language-SN220.txt` — Parcel name labels (one per line, 1-based index).
  Downloaded from `https://evlab.squarespace.com/s/allParcels-language-SN220.txt`.

## `cneuromod.all/`

DataLad superdataset cloned from `git@github.com:courtois-neuromod/cneuromod.all.git`.
Two subdatasets are used:

- `cneuromod.all/langlocalizer/bids/` — BIDS dataset with task event files (aliceEn, aliceFr, listening, reading).
- `cneuromod.all/langlocalizer/fmriprep/` — fMRIPrep preprocessed BOLD images, brain masks, and confound files.
- `cneuromod.all/hcptrt/bids/` — BIDS dataset with HCP task event files (motor, wm, gambling, social, language, relational, emotion).
- `cneuromod.all/hcptrt/fmriprep/` — fMRIPrep preprocessed files for hcptrt.

Each dataset's statistical model (contrasts, event transformations, file patterns) is described in the
corresponding `models/{dataset}.json` BIDS Stats Models 1.0.0 file.

📝 All data files in this folder are **ignored by Git** (see `.gitignore`).
