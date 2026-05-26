import json
from pathlib import Path


class ModelSpec:
    """Minimal parser for BIDS Stats Models 1.0.0 JSON files.

    Reads per-task Run-level and Subject-level Nodes plus the Extensions block
    used by this pipeline (FROIContrasts, FilePatterns, HasRunDimension).

    Transformation instructions supported:
      {"Name": "Replace", "Input": ["trial_type"], "Map": {...}}
      {"Name": "Drop",    "Input": ["trial_type"], "Values": [...]}
    """

    def __init__(self, path):
        with open(path) as f:
            self._data = json.load(f)
        self._nodes = self._data.get("Nodes", [])

    # ------------------------------------------------------------------
    # Node accessors
    # ------------------------------------------------------------------

    def tasks(self):
        return list(self._data.get("Input", {}).get("task", []))

    def _find_node(self, level, task):
        for node in self._nodes:
            if node.get("Level") != level:
                continue
            tasks_for_node = node.get("Filter", {}).get("task", [])
            if task in tasks_for_node:
                return node
        raise KeyError(f"No {level}-level node found for task '{task}'")

    def run_node(self, task):
        return self._find_node("Run", task)

    def subject_node(self, task):
        return self._find_node("Subject", task)

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def transformations(self, task):
        """Return Transformations.Instructions list for task's Run node."""
        node = self.run_node(task)
        return node.get("Transformations", {}).get("Instructions", [])

    # ------------------------------------------------------------------
    # Contrasts
    # ------------------------------------------------------------------

    def run_contrasts(self, task):
        """Return Contrasts list from the Run-level node for task."""
        return self.run_node(task).get("Contrasts", [])

    def subject_contrasts(self, task):
        """Return contrast names to carry to the subject level."""
        return [c["Name"] for c in self.subject_node(task).get("Contrasts", [])]

    # ------------------------------------------------------------------
    # Extensions
    # ------------------------------------------------------------------

    def _ext(self):
        return self._data.get("Extensions", {})

    def froi_contrasts(self, task):
        return self._ext().get("FROIContrasts", {}).get(task, [])

    def file_patterns(self):
        return self._ext().get("FilePatterns", {})

    def has_run_dimension(self):
        """True if sessions contain multiple numbered runs (e.g. hcptrt)."""
        return bool(self._ext().get("HasRunDimension", False))
