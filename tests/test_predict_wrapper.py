"""CPU tests of release validation and delegation, without loading model weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "tools/predict.py"
SPEC = importlib.util.spec_from_file_location("dev_predict", SOURCE)
predict = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predict)


class WrapperTests(unittest.TestCase):
    def test_identity_checks_bytes_and_rejects_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            file = directory / "weights"
            file.write_bytes(b"original")
            identities = {"weights": {"bytes": 8, "sha256": predict.sha256(file)}}
            predict.verify_files(directory, identities)
            file.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                predict.verify_files(directory, identities)

    def test_path_escape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside the snapshot"):
            predict.verify_files(Path.cwd(), {"../outside": {"bytes": 0, "sha256": ""}})

    def test_model_sees_no_gold_and_temperature_is_only_delegated(self):
        row = {"state": "toy state", "question": {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}}, "target": "yes", "label": "yes", "id": "toy-only"}

        class Fake:
            temperature = 0.9982566635954637

            def predict(self, rows, **kwargs):
                self.seen, self.kwargs = rows, kwargs
                return [[0.7, 0.3]]

        model = Fake()
        with patch.object(predict, "_answer", side_effect=lambda question, probabilities: {"probabilities": probabilities}):
            output = predict.predict_rows(model, [row], batch_size=1)
        self.assertEqual(set(model.seen[0]), {"state", "question"})
        self.assertEqual(model.kwargs, {"batch_size": 1})
        self.assertEqual(output, [{"probabilities": [0.7, 0.3]}])
        self.assertEqual(model.temperature, 0.9982566635954637)

    def test_empty_or_malformed_decisions_are_rejected(self):
        for value in ([], None, {"state": "no question"}):
            with self.assertRaises(ValueError):
                predict.normalize_rows(value)


if __name__ == "__main__":
    unittest.main()
