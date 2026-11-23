import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from human_annotations import analyze_annotations


class HumanAnnotationTests(unittest.TestCase):
    def test_agreement_report_uses_joint_human_labels_and_evidence_sets(self):
        fields = [
            "annotator_1_verdict",
            "annotator_1_evidence_passage_ids",
            "annotator_1_negative_category",
            "annotator_2_verdict",
            "annotator_2_evidence_passage_ids",
            "annotator_2_negative_category",
            "candidate_passages_json",
        ]
        rows = [
            {
                "annotator_1_verdict": "SUPPORTS",
                "annotator_1_evidence_passage_ids": "p1;p2",
                "annotator_1_negative_category": "",
                "annotator_2_verdict": "SUPPORTS",
                "annotator_2_evidence_passage_ids": "p2",
                "annotator_2_negative_category": "",
                "candidate_passages_json": "x" * 131_073,
            },
            {
                "annotator_1_verdict": "NOT_ENOUGH_INFORMATION",
                "annotator_1_evidence_passage_ids": "",
                "annotator_1_negative_category": "same_topic_insufficient_evidence",
                "annotator_2_verdict": "NOT_ENOUGH_INFORMATION",
                "annotator_2_evidence_passage_ids": "",
                "annotator_2_negative_category": "same_topic_insufficient_evidence",
                "candidate_passages_json": "x" * 131_073,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            annotation_path = Path(temp_dir) / "annotations.csv"
            report_path = Path(temp_dir) / "report.json"
            with annotation_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            analyze_annotations(annotation_path, report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["jointly_labeled_rows"], 2)
        self.assertEqual(report["cohen_kappa"], 1.0)
        self.assertEqual(report["evidence_passage_set_jaccard_mean"], 0.5)
        self.assertEqual(report["negative_category_pair_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
