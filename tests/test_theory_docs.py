"""Pin the documented claims of the philosophy layer (theory/ERROR_ATTRIBUTION_THEORY.md).

    python -m unittest discover -s tests -v

These are **documentation-consistency** tests. They fail when a claim written in the
docs stops being true: the disclaimer sentence is deleted, an entry-point link breaks,
the ethics ladder is renumbered, the attribution fork stops being binary, a glossary
term disappears, or the position note starts claiming empirical status.

They do **not** test the experiments (that is test_plastic_self.py and
test_multi_embodiment.py), and they deliberately do **not** pin the row count of tables
that are meant to grow (e.g. the 难题速查 lists). Counts are pinned only where the count
*is* the claim: the 0–6 ladder.

Variant self-check (the suite must not be vacuous): copy this file to a temp dir, invert
one assertion (e.g. `assertEqual(len(rows), 2)` -> `assertEqual(len(rows), 3)`), run
`python -m unittest discover -s <tmpdir>` and confirm a non-zero exit.
"""

from __future__ import annotations

import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
THEORY_DOC = REPO / "theory" / "ERROR_ATTRIBUTION_THEORY.md"
THEORY = REPO / "theory" / "THEORY.md"
QUESTIONS = REPO / "theory" / "QUESTIONS.md"
README = REPO / "README.md"
LANDING = REPO / "index.html"

DISCLAIMER_ZH = "这不是真理，这是一个模型。名 ≠ 道。"
BLOCK_URL = (
    "https://github.com/yiheweigui82/rsmeh/blob/main/theory/ERROR_ATTRIBUTION_THEORY.md"
)
LADDER = ["民", "唯我", "亲我", "群我", "类我", "生我", "无我"]
GLOSSARY_TERMS = [
    "道",
    "误差",
    "不可消除的误差",
    "归因于内",
    "归因于外",
    "我",
    "民",
    "觉醒",
    "迷失",
    "无我",
    "伦理空间",
    "持续回指",
]


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def table_rows(md: str, header_cell: str) -> list[list[str]]:
    """Return the body rows of the first markdown table whose header row holds header_cell."""
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header_cell not in cells:
            continue
        rows: list[list[str]] = []
        for nxt in lines[i + 1:]:
            if not nxt.strip().startswith("|"):
                break
            row = [c.strip() for c in nxt.strip().strip("|").split("|")]
            if set("".join(row)) <= set("-: "):  # separator row
                continue
            rows.append(row)
        return rows
    raise AssertionError(f"markdown table with header cell {header_cell!r} not found")


class TestDisclaimer(unittest.TestCase):
    def test_disclaimer_is_verbatim_in_the_theory_doc(self):
        self.assertIn(DISCLAIMER_ZH, read(THEORY_DOC))

    def test_disclaimer_reaches_readme_and_landing_page(self):
        self.assertIn(DISCLAIMER_ZH, read(README))
        self.assertIn(DISCLAIMER_ZH, read(LANDING))

    def test_disclaimer_sits_above_the_core_statement_in_readme(self):
        md = read(README)
        self.assertLess(
            md.index(DISCLAIMER_ZH),
            md.index("生命最初预测世界"),
            "免责声明必须排在仓库核心陈述之前",
        )


class TestEntryPoints(unittest.TestCase):
    def test_readme_links_the_doc_through_the_rendered_view(self):
        self.assertIn(BLOCK_URL, read(README))

    def test_landing_page_card_points_at_the_doc(self):
        self.assertIn(BLOCK_URL, read(LANDING))

    def test_repo_structure_table_lists_the_doc(self):
        paths = [r[0] for r in table_rows(read(README), "路径")]
        self.assertIn("`theory/ERROR_ATTRIBUTION_THEORY.md`", paths)


class TestEthicsLadder(unittest.TestCase):
    def test_ladder_is_exactly_seven_levels_labelled_zero_to_six(self):
        rows = table_rows(read(THEORY_DOC), "层次")
        self.assertEqual([r[0] for r in rows], [str(i) for i in range(7)])
        self.assertEqual([r[1] for r in rows], LADDER)

    def test_ladder_runs_from_min_to_wuwo_ending_at_dao(self):
        rows = table_rows(read(THEORY_DOC), "层次")
        self.assertEqual(rows[0][1], "民")
        self.assertEqual(rows[-1][1], "无我")
        self.assertIn("道", rows[-1][4])


class TestAttributionFork(unittest.TestCase):
    def test_fork_is_binary_and_has_two_distinct_outcomes(self):
        rows = table_rows(read(THEORY_DOC), "归因方向")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "归因于内")
        self.assertEqual(rows[0][1], "“我”诞生")
        self.assertIn("觉醒", rows[0][2])
        self.assertEqual(rows[1][0], "归因于外")
        self.assertEqual(rows[1][1], "神明诞生")
        self.assertIn("放弃觉醒", rows[1][2])


class TestGlossary(unittest.TestCase):
    def test_glossary_defines_the_load_bearing_terms(self):
        terms = [r[0] for r in table_rows(read(THEORY_DOC), "术语")]
        missing = [t for t in GLOSSARY_TERMS if t not in terms]
        self.assertEqual(missing, [], f"术语表缺少这些词条：{missing}")


class TestPositionNote(unittest.TestCase):
    def test_position_note_declines_empirical_status(self):
        # strip emphasis markers: the claim is about the prose, not about the bolding
        md = read(THEORY_DOC).replace("**", "")
        self.assertIn("哲学 / 规范性推论层，不是实验结论", md)
        self.assertIn("不能被本仓库的实验证实", md)
        self.assertIn("Level 4", md)
        self.assertIn("误读", md)

    def test_doc_is_cross_referenced_from_theory_and_questions(self):
        self.assertIn("ERROR_ATTRIBUTION_THEORY.md", read(THEORY))
        self.assertIn("ERROR_ATTRIBUTION_THEORY.md", read(QUESTIONS))


if __name__ == "__main__":
    unittest.main()
