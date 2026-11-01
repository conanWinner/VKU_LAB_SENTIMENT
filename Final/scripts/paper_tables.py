"""Write the LaTeX tables of the paper from files in `results/`, so no number is typed by hand.

    python -m absa.dataset_stats --data-dir data/Final_Data
    python scripts/paper_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"

ASPECTS = {
    "ky_nang_giang_day": ("Teaching Skill", "lesson organization and teaching methods"),
    "hanh_vi": ("Behavior", "lecturer's attitude and communication"),
    "de_xuat": ("Suggestion", "students' suggestions and wishes"),
    "bai_tap": ("Assignment", "assignments, their number and types"),
    "chuong_trinh_hoc": ("Curriculum", "curriculum and courses"),
    "kien_thuc": ("Knowledge", "lecturer's subject knowledge"),
    "kinh_nghiem": ("Experience", "practical experience in lectures"),
    "cung_cap_tai_lieu": ("Material Provision", "textbooks and learning materials"),
    "thiet_bi_day_hoc": ("Teaching Equipment", "classrooms, projectors, fans, lights"),
    "cham_diem": ("Grading", "grading activities"),
    "noi_chung": ("General", "other or general comments"),
}


def number(value: int) -> str:
    return f"{value:,}"


def table_aspects(stats: dict) -> str:
    rows = []
    per_aspect = stats["all"]["per_aspect"]
    for key, (name, description) in ASPECTS.items():
        row = per_aspect[key]
        total = sum(row.values())
        rows.append(f"{name} & {description} & {number(row['positive'])} & {number(row['negative'])} & {number(row['neutral'])} & {number(total)} \\\\")
    polarity = stats["all"]["polarity"]
    total = (f"Total & & {number(polarity['positive'])} & {number(polarity['negative'])} & {number(polarity['neutral'])} & {number(stats['all']['pairs'])} \\\\")
    return "\n".join([
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "Aspect & Scope & Pos. & Neg. & Neu. & Total \\\\", "\\midrule", *rows, "\\midrule", total, "\\bottomrule", "\\end{tabular}", "",
    ])


def table_splits(stats: dict) -> str:
    rows = []
    for name, row in [("Train", stats["splits"]["train"]), ("Validation", stats["splits"]["validation"]), ("Test", stats["splits"]["test"]), ("All", stats["all"])]:
        rows.append(f"{name} & {number(row['sentences'])} & {number(row['pairs'])} & {row['pairs_per_sentence']:.2f} & {number(row['multi_aspect_sentences'])} & {row['mean_tokens']:.1f} \\\\")
    return "\n".join([
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Split & Sentences & Pairs & Pairs/sent. & Multi-aspect & Tokens \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "",
    ])


if __name__ == "__main__":
    stats = json.loads((ROOT / "results" / "dataset_stats.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aspects.tex").write_text(table_aspects(stats), encoding="utf-8")
    (OUT / "splits.tex").write_text(table_splits(stats), encoding="utf-8")
    print("wrote", OUT, f"(dataset {stats['dataset_hash'][:8]})")
