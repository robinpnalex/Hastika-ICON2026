"""Fast repository contracts that require no model download or GPU."""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hastika.common.paths import PROJECT_ROOT, RAW_DATA_DIR, RUNS_DIR
from hastika.common.preprocessing import clean, dedupe_index
from hastika.common.submission import INPUTS, VALID
from hastika.models.baseline_svm import SPLIT_SEED


class RepositoryStructureTests(unittest.TestCase):
    def test_canonical_paths(self):
        self.assertEqual(PROJECT_ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(RAW_DATA_DIR, PROJECT_ROOT / "data" / "raw")
        self.assertEqual(RUNS_DIR, PROJECT_ROOT / "artifacts" / "runs")

    def test_raw_data_contract(self):
        expected_rows = {
            "binary_train.csv": 6446,
            "binary_validation_inputs.csv": 806,
            "multiclass_train.csv": 3159,
            "multiclass_validation_inputs.csv": 395,
        }
        for name, rows in expected_rows.items():
            with self.subTest(name=name):
                frame = pd.read_csv(RAW_DATA_DIR / name)
                self.assertEqual(len(frame), rows)
                self.assertIn("id", frame.columns)
                self.assertIn("Comment", frame.columns)

    def test_fixed_task_b_holdout_is_reproducible(self):
        frame = pd.read_csv(RAW_DATA_DIR / "multiclass_train.csv")
        keep = dedupe_index(
            frame["Comment"].tolist(), frame["Hate Category"].tolist(), "task B"
        )
        frame = frame.iloc[keep].reset_index(drop=True)
        labels = [
            "Gender", "Geo-political", "Others", "Political", "Religion", "Violence"
        ]
        y = frame["Hate Category"].map(labels.index).to_numpy()
        _, val_idx = train_test_split(
            np.arange(len(y)), test_size=0.15, stratify=y, random_state=SPLIT_SEED
        )
        stored = pd.read_csv(
            PROJECT_ROOT / "data" / "derived" / "task_b_holdout" / "val_inputs.csv"
        )
        self.assertEqual(stored["id"].tolist(), frame.iloc[val_idx]["id"].tolist())

    def test_submission_contracts_reference_raw_inputs(self):
        self.assertEqual(set(INPUTS), {"a", "b"})
        self.assertEqual(VALID["a"], {"Hate", "Non-Hate"})
        self.assertEqual(len(VALID["b"]), 6)
        for filename in INPUTS.values():
            self.assertTrue((RAW_DATA_DIR / filename).is_file())

    def test_task_b_notebooks_are_valid_json(self):
        notebooks = sorted((PROJECT_ROOT / "notebooks" / "task_b").glob("*.ipynb"))
        self.assertGreaterEqual(len(notebooks), 1)
        for notebook in notebooks:
            with self.subTest(notebook=notebook.name):
                payload = json.loads(notebook.read_text())
                self.assertEqual(payload["nbformat"], 4)
                self.assertTrue(payload["cells"])

    def test_basic_cleaning(self):
        self.assertEqual(clean("hello<br>world"), "hello world")


if __name__ == "__main__":
    unittest.main()
