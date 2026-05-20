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
Only the `langlocalizer` subdataset is used by this project:

- `cneuromod.all/langlocalizer/bids/` — BIDS dataset with task event files.
- `cneuromod.all/langlocalizer/fmriprep/` — fMRIPrep preprocessed BOLD images,
  brain masks, and confound files.

📝 All data files in this folder are **ignored by Git** (see `.gitignore`).
