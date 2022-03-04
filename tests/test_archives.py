import csv
import unittest
import os
import os.path as op

from main import main


TEST_DATA_DIR = op.join(op.dirname(__file__), "data")
TEST_OUTPUT_DIR = op.join(op.dirname(__file__), "output")
TEST_REF_DIR = op.join(op.dirname(__file__), "ref_output")


class TestArchives(unittest.TestCase):
    archives = ["NHArchief", "GrArchief"]
    def test_archive_output(self) -> None:
        # GIVEN
        # The photo's and newspapers as given in the data directory
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

        for archive in self.archives:
            with self.subTest(msg=f"Testing archive {archive}"):
                # WHEN
                main(op.join(TEST_DATA_DIR, archive), TEST_OUTPUT_DIR)

                # THEN
                self._compare_csv(
                    op.join(TEST_REF_DIR, f"match_links_{archive}.csv"),
                    op.join(TEST_OUTPUT_DIR, "match_links.csv")
                )

    def _compare_csv(self, ref_file, out_file):
        with open(ref_file) as ref_f:
            ref_reader = csv.DictReader(ref_f)
            with open(out_file) as out_f:
                out_reader = csv.DictReader(out_f)
                ref = list(ref_reader)
                out = list(out_reader)
                self.assertEqual(ref, out)
