"""Annotation conventions and complete-image coverage are scientific acceptance rules."""
import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from microcture.tasks.task3.atlas import read_reference, audit_gene_rows, check_pages, page_html


class AnnotationTests(unittest.TestCase):
    def reference(self, root, chrom="NC_000913.3"):
        path = root/"reference.gff"
        path.write_text(f"##gff-version 3\n##sequence-region {chrom} 1 1000\n"
                        f"{chrom}\tRefSeq\tgene\t190\t255\t.\t+\t.\tID=gene-b0001;gene=thrL;locus_tag=b0001;gene_synonym=oldThr\n"
                        f"{chrom}\tRefSeq\tpseudogene\t500\t700\t.\t-\t.\tID=gene-b0002;gene=other;locus_tag=b0002\n")
        return path

    def test_reference_converts_closed_endpoints_once_and_keeps_pseudogenes(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            genes = read_reference(self.reference(Path(folder)), "NC_000913.3", 1000)
            self.assertEqual((genes[0]["start"], genes[0]["end"]), (189, 255))
            self.assertEqual(genes[1]["feature_type"], "pseudogene")

    def test_wrong_accession_or_sequence_length_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            with self.assertRaisesRegex(ValueError, "identity/length"):
                read_reference(self.reference(Path(folder)), "NC_000913.2", 1000)

    def test_alias_closed_coordinates_differ_from_halfopen_and_strand_mismatch(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            genes = read_reference(self.reference(Path(folder)), "NC_000913.3", 1000)
            src = [{"gene": "oldThr", "start": "190", "end": "255", "strand": "+"},
                   {"gene": "thrL", "start": "189", "end": "255", "strand": "+"},
                   {"gene": "thrL", "start": "190", "end": "255", "strand": "-"},
                   {"gene": "unmapped", "start": "190", "end": "255", "strand": "+"}]
            result = audit_gene_rows(src, genes)
            self.assertEqual([r["status"] for r in result], ["exact_1_closed", "exact_0_halfopen", "name_match_coordinates_differ", "name_not_found"])

    def test_duplicate_source_names_and_uncertain_rows_are_never_removed(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            genes = read_reference(self.reference(Path(folder)), "NC_000913.3", 1000)
            src = [{"gene": "thrL", "start": "190", "end": "255", "strand": "+"}]*2
            result = audit_gene_rows(src, genes)
            self.assertEqual(len(result), 2)
            self.assertEqual([r["source_row"] for r in result], [2, 3])


class CompleteTrackTests(unittest.TestCase):
    def test_csv_string_coordinates_render_as_integer_genome_positions(self):
        page = page_html(464, {"start": "4640000", "end": "4641652", "file": "tracks/4640000_4641652.png"})
        self.assertIn("4,640,000–4,641,652", page)
        self.assertIn('data-page="464"', page)

    def fixture(self, root):
        (root/"tracks").mkdir()
        result = []
        for start, end in [(0, 10000), (10000, 16520)]:
            name = f"tracks/{start:07d}_{end:07d}.png"
            Image.new("RGB", (600, 600), "white").save(root/name)
            result.append({"start": str(start), "end": str(end), "file": name})
        return result

    def test_partial_terminal_page_is_part_of_complete_coverage(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            root = Path(folder);check_pages(self.fixture(root), root, 16520, 10000)

    def test_full_csv_without_actual_image_fails(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            root = Path(folder);pages = self.fixture(root);(root/pages[1]["file"]).unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                check_pages(pages, root, 16520, 10000)

    def test_shifted_coordinate_or_duplicate_page_fails(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            root = Path(folder);pages = self.fixture(root);pages[1]["start"] = "9999"
            with self.assertRaisesRegex(ValueError, "coordinate/path"):
                check_pages(pages, root, 16520, 10000)

    def test_unindexed_image_cannot_be_hidden(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            root = Path(folder);pages = self.fixture(root)
            Image.new("RGB", (600, 600)).save(root/"tracks/extra.png")
            with self.assertRaisesRegex(ValueError, "Unindexed"):
                check_pages(pages, root, 16520, 10000)


if __name__ == "__main__":
    unittest.main()
