import unittest
import json
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd

from data_integrity import split_real_records
from data_loader import _read_jarvis_archive, _require_real_identifiers
from monte_carlo import generate_synthetic
from config import ALL_DESCRIPTORS, TARGET_COLUMN
from feature_engineering import engineer_features
from explainability import bootstrap_uncertainty
from sklearn.ensemble import RandomForestRegressor


class ProvenanceTests(unittest.TestCase):
    def test_jarvis_figshare_zip_is_read_without_extraction(self):
        records = [{"jid": "JVASP-1", "optb88vdw_bandgap": 1.2}]
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.dirname(__file__))) as tmp:
            path = os.path.join(tmp, "jarvis.zip")
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("jdft_3d-9-24-2025.json", json.dumps(records))
            self.assertEqual(_read_jarvis_archive(path), records)

    def test_unrecognized_archive_fails_clearly(self):
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.dirname(__file__))) as tmp:
            path = os.path.join(tmp, "jarvis.zip")
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("unrelated.json", "[]")
            with self.assertRaisesRegex(ValueError, "Expected one jdft_3d"):
                _read_jarvis_archive(path)

    def test_surrogate_ids_rejected_as_real_reference(self):
        frame = pd.DataFrame({"jid": ["SURR-00001"]})
        with self.assertRaisesRegex(ValueError, "synthetic or missing IDs"):
            _require_real_identifiers(frame, "test fixture")

    def test_missing_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "no JARVIS material IDs"):
            _require_real_identifiers(pd.DataFrame({"formula": ["Si"]}), "test")

    def test_real_split_is_reproducible_and_disjoint(self):
        records = pd.DataFrame({"jid": [f"JVASP-{i}" for i in range(50)]})
        train_a, test_a = split_real_records(records, random_state=17)
        train_b, test_b = split_real_records(records, random_state=17)
        self.assertEqual(train_a.jid.tolist(), train_b.jid.tolist())
        self.assertEqual(test_a.jid.tolist(), test_b.jid.tolist())
        self.assertFalse(set(train_a.jid) & set(test_a.jid))

    def test_duplicate_material_ids_rejected(self):
        records = pd.DataFrame({"jid": ["JVASP-1", "JVASP-1"]})
        with self.assertRaisesRegex(ValueError, "present and unique"):
            split_real_records(records)


class MonteCarloTests(unittest.TestCase):
    def test_gaussian_copula_sampling_is_seeded(self):
        rng = np.random.default_rng(4)
        n = 220
        values = rng.normal(size=(n, len(ALL_DESCRIPTORS) + 1))
        frame = pd.DataFrame(values, columns=ALL_DESCRIPTORS + [TARGET_COLUMN])
        frame["ehull"] = np.abs(frame["ehull"])
        frame["density"] = np.abs(frame["density"]) + 0.2
        frame["mean_atomic_mass"] = np.abs(frame["mean_atomic_mass"]) + 1
        frame["mean_covalent_radius"] = np.abs(frame["mean_covalent_radius"]) + 0.3
        frame[["a", "b", "c"]] = frame[["a", "b", "c"]].abs() + 2
        frame["volume"] = frame["volume"].abs() + 50
        frame[TARGET_COLUMN] = frame[TARGET_COLUMN].abs() + 0.01
        first = generate_synthetic(frame, n_samples=4, seed=123)
        second = generate_synthetic(frame, n_samples=4, seed=123)
        pd.testing.assert_frame_equal(first, second)

    def test_zero_density_does_not_create_extreme_derived_feature(self):
        row = {
            "mean_atomic_mass": 1.008, "density": 0.0,
            "formation_energy_peratom": -0.1, "ehull": 0.0,
            "mean_covalent_radius": 0.3, "a": 20.0, "b": 20.0,
            "c": 20.0, "volume": 8000.0, "n_atoms": 1,
            "n_elements": 1, "mean_atomic_number": 1,
            "mean_electronegativity": 2.2, "std_atomic_mass": 0.0,
            "std_electronegativity": 0.0, "std_covalent_radius": 0.0,
            "max_electronegativity": 2.2, "min_electronegativity": 2.2,
            TARGET_COLUMN: 5.137,
        }
        result = engineer_features(pd.DataFrame([row]), verbose=False)
        self.assertTrue(np.isfinite(result["mass_density_ratio"].iloc[0]))
        self.assertEqual(result["mass_density_ratio"].iloc[0], 0.0)

    def test_bootstrap_intervals_use_training_only_oob_residuals(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(100, 3))
        y = x[:, 0] * 2.0 + rng.normal(scale=0.5, size=len(x))
        result = bootstrap_uncertainty(
            RandomForestRegressor,
            {"n_estimators": 10, "random_state": 3, "n_jobs": 1},
            x[:80], y[:80], x[80:], n_bootstraps=8, confidence=0.9,
        )
        self.assertGreater(result["calibration_residual_count"], 0)
        self.assertTrue(np.all(result["lower"] <= result["mean"]))
        self.assertTrue(np.all(result["upper"] >= result["mean"]))
        self.assertGreater(float(np.mean(result["upper"] - result["lower"])), 0.0)


if __name__ == "__main__":
    unittest.main()
