"""utils/compare_subject_outputs.py: tells identical runs from differing ones."""

import numpy as np
import pandas as pd

from utils.compare_subject_outputs import compare_csv, compare_subject_dirs


def write(folder, rel, df):
    path = folder / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


FRAME = pd.DataFrame({"model_descr": ["a"] * 3, "trial_type": ["x", "y", None], "evidence_face": [0.1, 0.5, np.nan], "correct": [True, False, True]})


def test_identical_files_match_even_with_a_different_model_name(tmp_path):
    write(tmp_path / "A", "decoding/s_decoding_results.csv", FRAME)
    write(tmp_path / "B", "decoding/s_decoding_results.csv", FRAME.assign(model_descr="b"))
    assert compare_subject_dirs(str(tmp_path / "A"), str(tmp_path / "B")) == {"decoding/s_decoding_results.csv": []}


def test_numeric_difference_beyond_tolerance_is_reported_and_within_it_is_not(tmp_path):
    write(tmp_path / "A", "model/s_cv_results.csv", FRAME)
    write(tmp_path / "B", "model/s_cv_results.csv", FRAME.assign(evidence_face=FRAME["evidence_face"] + 1e-12))
    assert compare_subject_dirs(str(tmp_path / "A"), str(tmp_path / "B"))["model/s_cv_results.csv"] == []
    write(tmp_path / "C", "model/s_cv_results.csv", FRAME.assign(evidence_face=FRAME["evidence_face"] + 1e-3))
    problems = compare_subject_dirs(str(tmp_path / "A"), str(tmp_path / "C"))["model/s_cv_results.csv"]
    assert len(problems) == 1 and "evidence_face" in problems[0]


def test_text_differences_row_counts_and_missing_files_are_reported(tmp_path):
    write(tmp_path / "A", "model/m.csv", FRAME)
    write(tmp_path / "B", "model/m.csv", FRAME.assign(trial_type=["x", "z", None]))
    assert "trial_type: 1 value(s) differ" in compare_subject_dirs(str(tmp_path / "A"), str(tmp_path / "B"))["model/m.csv"]
    write(tmp_path / "C", "model/m.csv", FRAME.iloc[:2])
    assert compare_subject_dirs(str(tmp_path / "A"), str(tmp_path / "C"))["model/m.csv"] == ["3 rows vs 2 rows"]
    write(tmp_path / "A", "test/only_a.csv", FRAME)
    assert compare_subject_dirs(str(tmp_path / "A"), str(tmp_path / "B"))["test/only_a.csv"] == ["missing from the new run"]


def test_a_column_that_is_nan_in_both_runs_is_not_a_difference(tmp_path):
    write(tmp_path / "A", "model/m.csv", FRAME)
    write(tmp_path / "B", "model/m.csv", FRAME)
    assert compare_csv(str(tmp_path / "A" / "model/m.csv"), str(tmp_path / "B" / "model/m.csv")) == []
