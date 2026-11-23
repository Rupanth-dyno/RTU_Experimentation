import unittest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.run_experiments import CachedLLMVerifier, evaluate_strategy
from scripts.experiment_utils import (
    connected_groups,
    make_grouped_split,
    paired_cluster_bootstrap_difference,
    paired_cluster_permutation,
    summarize_metrics,
)


def instance(instance_id, pair_id, citing_id, source_id, label):
    return {
        "instance_id": instance_id,
        "pair_id": pair_id,
        "citing_arxiv_id": citing_id,
        "source_arxiv_id": source_id,
        "label": label,
        "ground_truth": "SUPPORTS" if label else "NOT_ENOUGH_INFORMATION",
    }


class GroupedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for pair in range(20):
            self.rows.extend(
                [
                    instance(f"{pair}-p", pair, f"c{pair}", f"s{pair}", 1),
                    instance(f"{pair}-n", pair, f"c{pair}", f"d{pair}", 0),
                ]
            )

    def test_group_split_is_deterministic_and_prevents_identifier_leakage(self):
        first = make_grouped_split(self.rows, seed=7)
        second = make_grouped_split(self.rows, seed=7)
        self.assertEqual(first["splits"], second["splits"])
        self.assertEqual(sum(map(len, first["splits"].values())), len(self.rows))
        pair_splits = {}
        citing_splits = {}
        source_splits = {}
        row_by_id = {row["instance_id"]: row for row in self.rows}
        for split_name, ids in first["splits"].items():
            for instance_id in ids:
                row = row_by_id[instance_id]
                for mapping, key in (
                    (pair_splits, row["pair_id"]),
                    (citing_splits, row["citing_arxiv_id"]),
                    (source_splits, row["source_arxiv_id"]),
                ):
                    if key in mapping:
                        self.assertEqual(mapping[key], split_name)
                    mapping[key] = split_name

    def test_shared_source_merges_groups(self):
        rows = [
            instance("a-p", 1, "c1", "same-source", 1),
            instance("a-n", 1, "c1", "d1", 0),
            instance("b-p", 2, "c2", "same-source", 1),
            instance("b-n", 2, "c2", "d2", 0),
        ]
        group_ids, _ = connected_groups(rows)
        self.assertEqual(group_ids[0], group_ids[2])

    def test_metrics_and_paired_test_include_correct_shapes(self):
        logs = [
            {
                "instance_id": "p",
                "ground_truth": "SUPPORTS",
                "predicted": "SUPPORTS",
                "correct": True,
                "iterations_used": 1,
                "first_verdict": "SUPPORTS",
                "cache_hits": 0,
                "cache_misses": 1,
            },
            {
                "instance_id": "n",
                "ground_truth": "NOT_ENOUGH_INFORMATION",
                "predicted": "SUPPORTS",
                "correct": False,
                "iterations_used": 2,
                "first_verdict": "NOT_ENOUGH_INFORMATION",
                "cache_hits": 1,
                "cache_misses": 1,
            },
        ]
        summary = summarize_metrics("test", logs, {"p": "g1", "n": "g1"}, bootstrap_samples=20)
        self.assertEqual(summary["confusion_matrix"]["rows_true_columns_predicted"], [[0, 1], [0, 1]])
        self.assertEqual(summary["false_positive"], 1)
        self.assertIn("support_f1_cluster_bootstrap_95_ci", summary)
        self.assertEqual(summary["median_checks"], 1)
        self.assertEqual(summary["p95_checks"], 2)
        paired = paired_cluster_permutation(logs, logs, {"p": "g1", "n": "g1"})
        self.assertEqual(paired["two_sided_p_value"], 1.0)
        paired_ci = paired_cluster_bootstrap_difference(
            logs, logs, {"p": "g1", "n": "g1"}, repetitions=100
        )
        self.assertEqual(paired_ci["accuracy_difference_cluster_bootstrap_95_ci"], [0.0, 0.0])


class StructuredVerifierTests(unittest.TestCase):
    def test_structured_verdict_is_cached_with_a_configuration_bound_key(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.sqlite3"
            version = Mock()
            version.raise_for_status.return_value = None
            version.json.return_value = {"version": "0.5.0"}
            tags = Mock()
            tags.raise_for_status.return_value = None
            tags.json.return_value = {"models": [{"name": "qwen2.5:3b", "digest": "digest-a"}]}
            model_response = Mock()
            model_response.raise_for_status.return_value = None
            model_response.json.return_value = {"message": {"content": '{"verdict":"SUPPORTS"}'}}

            with patch("scripts.run_experiments.requests.get", side_effect=[version, tags]), patch(
                "scripts.run_experiments.requests.post", return_value=model_response
            ) as post:
                verifier = CachedLLMVerifier(
                    "http://localhost:11434", "qwen2.5:3b", cache_path, 5
                )
                verdict, latency, cache_hit, raw_answer, parse_status = verifier.verify(
                    "claim", "passage"
                )
                self.assertEqual(verdict, "SUPPORTS")
                self.assertGreaterEqual(latency, 0)
                self.assertFalse(cache_hit)
                self.assertEqual(raw_answer, '{"verdict":"SUPPORTS"}')
                self.assertEqual(parse_status, "ok")
                self.assertEqual(
                    verifier.verify("claim", "passage"),
                    ("SUPPORTS", 0.0, True, '{"verdict":"SUPPORTS"}', "ok"),
                )
                self.assertEqual(post.call_count, 1)
                self.assertEqual(
                    post.call_args.kwargs["json"]["format"]["properties"]["verdict"]["enum"],
                    ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFORMATION"],
                )
                verifier.db.close()

            changed_tags = Mock()
            changed_tags.raise_for_status.return_value = None
            changed_tags.json.return_value = {"models": [{"name": "qwen2.5:3b", "digest": "digest-b"}]}
            with patch("scripts.run_experiments.requests.get", side_effect=[version, changed_tags]), patch(
                "scripts.run_experiments.requests.post", return_value=model_response
            ) as post:
                verifier = CachedLLMVerifier(
                    "http://localhost:11434", "qwen2.5:3b", cache_path, 5
                )
                verifier.verify("claim", "passage")
                self.assertEqual(post.call_count, 1)
                verifier.db.close()

    def test_invalid_structured_output_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            version = Mock()
            version.raise_for_status.return_value = None
            version.json.return_value = {"version": "0.5.0"}
            tags = Mock()
            tags.raise_for_status.return_value = None
            tags.json.return_value = {"models": [{"name": "qwen2.5:3b", "digest": "digest-a"}]}
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"message": {"content": "NOT SUPPORTS"}}
            with patch("scripts.run_experiments.requests.get", side_effect=[version, tags]), patch(
                "scripts.run_experiments.requests.post", return_value=response
            ):
                verifier = CachedLLMVerifier(
                    "http://localhost:11434",
                    "qwen2.5:3b",
                    Path(directory) / "cache.sqlite3",
                    5,
                )
                result = verifier.verify("claim", "passage")
                self.assertEqual(result[0], "PARSE_FAILURE")
                self.assertEqual(result[3], "NOT SUPPORTS")
                self.assertEqual(result[4], "parse_failure")
                verifier.db.close()


class EvaluationLoggingTests(unittest.TestCase):
    def test_strategy_logs_prediction_correctness_and_check_cost(self):
        class FakeVerifier:
            config = {"model": "test"}

            def verify(self, claim, passage):
                return "SUPPORTS", 0.125, False, '{"verdict":"SUPPORTS"}', "ok"

        row = {
            "instance_id": "positive",
            "label": 1,
            "ground_truth": "SUPPORTS",
            "claim_context": "claim",
            "paragraphs": [{"text": "supporting passage"}],
        }
        result = evaluate_strategy(
            "test",
            [row],
            [[0]],
            [0.01],
            FakeVerifier(),
            1,
            {"positive": "component"},
            time.perf_counter(),
        )
        self.assertTrue(result["logs"][0]["correct"])
        self.assertEqual(result["logs"][0]["iterations_used"], 1)
        self.assertEqual(result["summary"]["true_positive"], 1)
        self.assertEqual(
            result["logs"][0]["passage_verifier_outputs"][0]["raw_answer"],
            '{"verdict":"SUPPORTS"}',
        )


if __name__ == "__main__":
    unittest.main()
