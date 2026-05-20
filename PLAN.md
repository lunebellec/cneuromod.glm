# Next Steps

## 1. Debug fetch on the cluster

Run `invoke fetch --subjects sub-01 --tasks listening` on a machine with Compute Canada access and verify:

- `datalad get` glob patterns resolve correctly inside `source_data/cneuromod.all/`
- If globs fail, switch to explicit per-file `datalad get` using `pathlib.glob()` on the installed metadata tree
- Confirm the ria siblings (`ria-beluga-storage`, `ria-sequoia-storage`) are enabled and accessible

## 2. Verify the path structure

The code assumes:

```
source_data/cneuromod.all/langlocalizer/bids/sub-{sub}/ses-{ses}/func/*task-{task}*_events.tsv
source_data/cneuromod.all/langlocalizer/fmriprep/sub-{sub}/ses-{ses}/func/*task-{task}*_part-mag_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
```

Check that the actual BIDS and fmriprep directory trees match these patterns. If not, update `FUNC_PATTERN` and `MASK_PATTERN` in `analysis/glm.py` and `find_sessions_for_subject()`.

## 3. Fix the run-glm sentinel check

`run_glm` currently skips a session if `contrast-int-degr_stat-z.nii.gz` exists, but the `reading` task produces `contrast-word-nonword_stat-z.nii.gz`. Fix: use the session directory itself as the sentinel (non-empty directory = done), or pick the first contrast from `TASK_CONTRASTS[task]`.

## 4. Run a proper smoke test

```bash
uv run invoke run-smoke
```

Expected: atlas downloads, one subject's GLM runs for one session, subject-level and fROI steps produce output, notebook saves a figure.

## 5. Add fROI visualization notebook

Create `notebooks/froi_results.ipynb` that reads `output_data/fedorenko_frois/` and plots the session-level and subject-level fROI masks per parcel using nilearn glass-brain views.

## 6. Add session-level fROI extraction (optional)

`analysis/froi.py` has `extract_session_frois()` but `run_froi` in `tasks.py` only calls `extract_subject_frois()`. Add a `run-froi-session` task if per-session fROI masks are needed.

## 7. Second-level GLM (optional)

If a group-level analysis across subjects is needed, add:
- `analysis/group_glm.py` with a `run_group_glm()` function using nilearn `SecondLevelModel`
- `run-group` and `clean-group` tasks in `tasks.py`
- Wire into the top-level `run` chain
