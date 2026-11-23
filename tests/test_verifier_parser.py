import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_experiments import parse_verifier_json


class VerifierParserTests(unittest.TestCase):
    def test_exact_labels(self):
        self.assertEqual(parse_verifier_json('{"verdict":"SUPPORTS"}'), ("SUPPORTS", "ok"))
        self.assertEqual(
            parse_verifier_json('{"verdict":"REFUTES"}'), ("REFUTES", "ok")
        )
        self.assertEqual(
            parse_verifier_json('{"verdict":"NOT_ENOUGH_INFORMATION"}'),
            ("NOT_ENOUGH_INFORMATION", "ok"),
        )

    def test_partial_and_contradictory_text_fail(self):
        self.assertEqual(parse_verifier_json("SUPPORTS ONLY PARTIALLY"), ("PARSE_FAILURE", "parse_failure"))
        self.assertEqual(parse_verifier_json("NOT SUPPORTS"), ("PARSE_FAILURE", "parse_failure"))
        self.assertEqual(
            parse_verifier_json('{"verdict":"SUPPORTS","verdict":"REFUTES"}'),
            ("PARSE_FAILURE", "parse_failure"),
        )

    def test_extra_keys_and_bad_json_fail(self):
        self.assertEqual(
            parse_verifier_json('{"verdict":"SUPPORTS","reason":"yes"}'),
            ("PARSE_FAILURE", "parse_failure"),
        )
        self.assertEqual(parse_verifier_json('["SUPPORTS"]'), ("PARSE_FAILURE", "parse_failure"))
        self.assertEqual(parse_verifier_json("{not json}"), ("PARSE_FAILURE", "parse_failure"))

if __name__ == "__main__":
    unittest.main()
