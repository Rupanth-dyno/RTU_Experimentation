import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_utils import (
    connected_groups,
    make_grouped_split,
    paired_cluster_bootstrap_difference,
    paired_cluster_permutation,
)


def row(instance_id, pair_id, citing_id, source_id, label):
    return {
        "instance_id": instance_id,
        "pair_id": pair_id,
        "citing_arxiv_id": citing_id,
        "source_arxiv_id": source_id,
        "label": label,
    }


class GroupedMetricTests(unittest.TestCase):
    def test_components_join_rows_by_pair_citing_and_source(self):
        rows = [
            row("p1", 1, "c1", "s1", 1),
            row("n1", 1, "c1", "d1", 0),
            row("p2", 2, "c2", "s1", 1),
            row("n2", 2, "c2", "d2", 0),
        ]
        group_ids, _ = connected_groups(rows)
        self.assertEqual(group_ids[0], group_ids[1])
        self.assertEqual(group_ids[0], group_ids[2])

    def test_split_is_deterministic_and_has_no_group_leakage(self):
        rows = []
        for pair in range(30):
            rows.extend([
                row(f"{pair}-p", pair, f"c{pair}", f"s{pair}", 1),
                row(f"{pair}-n", pair, f"c{pair}", f"d{pair}", 0),
            ])
        split = make_grouped_split(rows, seed=7)
        self.assertEqual(split["splits"], make_grouped_split(rows, seed=7)["splits"])
        row_by_id = {entry["instance_id"]: entry for entry in rows}
        seen = {}
        for split_name, ids in split["splits"].items():
            for instance_id in ids:
                entry = row_by_id[instance_id]
                for key in ("pair_id", "citing_arxiv_id", "source_arxiv_id"):
                    value = entry[key]
                    if (key, value) in seen:
                        self.assertEqual(seen[(key, value)], split_name)
                    seen[(key, value)] = split_name

    def test_paired_permutation_reports_group_unit(self):
        first = [{"instance_id": "p", "correct": True}]
        second = [{"instance_id": "p", "correct": False}]
        result = paired_cluster_permutation(first, second, {"p": "component"})
        self.assertEqual(result["groups"], 1)
        self.assertEqual(result["accuracy_difference"], 1.0)
        self.assertEqual(result["two_sided_p_value"], 1.0)

    def test_paired_bootstrap_reports_difference_interval(self):
        first = [{"instance_id": "p", "correct": True}]
        second = [{"instance_id": "p", "correct": False}]
        result = paired_cluster_bootstrap_difference(
            first, second, {"p": "component"}, repetitions=100
        )
        self.assertEqual(result["accuracy_difference_cluster_bootstrap_95_ci"], [1.0, 1.0])
        self.assertEqual(result["bootstrap_groups"], 1)


if __name__ == "__main__":
    unittest.main()
