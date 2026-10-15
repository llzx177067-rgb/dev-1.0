"""Small CPU tests of exact-case scoring and prediction artifact validation."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("score_bfcl", Path(__file__).resolve().parents[1] / "evaluation/score_bfcl.py")
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


def toy():
    answer = lambda choice: {"type": "choice", "choice": choice, "probabilities": {"yes": 0.8 if choice == "yes" else 0.2, "no": 0.2 if choice == "yes" else 0.8}}
    return {"schema": "dev-bfcl-predictions-1", "id": "toy-case", "request_id": "toy-request", "catalog_id": 1, "payload_sha256": "a" * 64, "track": "toy", "candidate_tools": ["alpha", "beta"], "field_order": ["tool_alpha", "tool_beta"], "option_order": {"tool_alpha": ["yes", "no"], "tool_beta": ["yes", "no"]}, "expected": {"tool_alpha": "yes", "tool_beta": "no"}, "answers": {score.QWEN: {"tool_alpha": answer("yes"), "tool_beta": answer("yes")}, score.DEV: {"tool_alpha": answer("yes"), "tool_beta": answer("no")}}}


class ScoreTests(unittest.TestCase):
    def test_case_exact_differs_from_field_accuracy(self):
        result = score.score_records([toy()])
        self.assertEqual(result["models"][score.QWEN]["field_accuracy"], 0.5)
        self.assertEqual(result["models"][score.QWEN]["case_exact_accuracy"], 0.0)
        self.assertEqual(result["models"][score.DEV]["case_exact_accuracy"], 1.0)
        self.assertEqual(result["paired_cases"]["dev_only_correct"], 1)

    def test_missing_candidate_is_rejected(self):
        row = toy()
        del row["answers"][score.DEV]["tool_beta"]
        with self.assertRaisesRegex(ValueError, "candidate field"):
            score.score_records([row])

    def test_invalid_probabilities_are_rejected(self):
        for probabilities in ({"yes": float("nan"), "no": 0}, {"yes": -0.1, "no": 1.1}, {"yes": 0.8, "no": 0.8}, {"yes": 1.0}):
            row = toy()
            row["answers"][score.QWEN]["tool_alpha"]["probabilities"] = probabilities
            with self.assertRaises(ValueError):
                score.score_records([row])

    def test_duplicate_case_or_different_model_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            score.score_records([toy(), toy()])
        row = toy()
        row["answers"].pop(score.DEV)
        with self.assertRaisesRegex(ValueError, "two published models"):
            score.score_records([row])


if __name__ == "__main__":
    unittest.main()
