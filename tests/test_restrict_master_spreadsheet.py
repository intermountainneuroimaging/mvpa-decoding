"""restrict_master_spreadsheet.py: balanced training-trial list. Synthetic data
only (tiny NIfTIs + events.tsv written to tmp_path), run end to end through the
real master-spreadsheet builder and the real training trial selection
(label_conditions_with_lag + load_trial_restriction)."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from utils.mvpa_common import label_conditions_with_lag, load_trial_restriction
from workflows import restrict_master_spreadsheet as rms
from workflows.generate_master_spreadsheet import process_events_file

TR = 2.0
N_VOLS = 200


def _q(column, value):
    return {"column": column, "match": "regex", "value": value}


CFG = {
    "seed": 7,
    "primary_groups": {"face": _q("trial_type", "face_.*"), "place": _q("trial_type", "place_.*"), "item": _q("trial_type", "item")},
    "secondary_groups": {"positive": _q("trial_type", ".*_pos"), "negative": _q("trial_type", ".*_neg")},
    "secondary_applies_to": ["face", "place"],
}


def write_run(root, sub, run, trials):
    """trials: list of (trial_type, duration_s); one every 8 s, plus a fixation row between."""
    d = root / f"sub-{sub}" / "func"
    d.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.zeros((3, 3, 3, N_VOLS), dtype=np.float32), np.eye(4))
    img.header.set_zooms((3.0, 3.0, 3.0, TR))
    bold = d / f"sub-{sub}_task-t_run-{run:02d}_bold.nii.gz"
    img.to_filename(str(bold))
    rows = []
    for i, (tt, dur) in enumerate(trials):
        onset = 2.0 + 8.0 * i
        rows.append({"onset": onset, "duration": dur, "trial_type": tt, "rt": f"{0.5 + i:g}"})
        rows.append({"onset": onset + dur, "duration": 1.0, "trial_type": "fixation", "rt": "n/a"})
    ev = d / f"sub-{sub}_task-t_run-{run:02d}_events.tsv"
    pd.DataFrame(rows).to_csv(ev, sep="\t", index=False)
    return str(bold), str(ev)


def make_subject(root, sub, per_run, dur=4.0):
    """per_run: list (one per run) of {trial_type: count}; trials interleaved in a fixed order."""
    for run, counts in enumerate(per_run, start=1):
        trials = []
        pools = {k: v for k, v in counts.items()}
        while any(pools.values()):
            for k in list(pools):
                if pools[k]:
                    pools[k] -= 1
                    trials.append((k, dur))
        write_run(root, sub, run, trials)


def build_master(root, subjects):
    tables = []
    for sub in subjects:
        for ev in sorted((root / f"sub-{sub}" / "func").glob("*_events.tsv")):
            tables.append(process_events_file(str(ev), str(root)))
    return pd.concat(tables, ignore_index=True).astype(str).fillna("")


def restrict(tmp_path, master, cfg=CFG, lag=0.0):
    return rms.restrict_trials(master, cfg, lag)


# the example from the request: 36 face (18+18), 36 place (18+18), 12 item
EXAMPLE = [
    {"face_pos": 9, "face_neg": 9, "place_pos": 9, "place_neg": 9, "item": 6},
    {"face_pos": 9, "face_neg": 9, "place_pos": 9, "place_neg": 9, "item": 6},
]


def n_trials(labeled):
    """distinct trials per regressor_label (trial_index restarts in every run)"""
    return labeled.drop_duplicates(["regressor_label", "boldfile", "trial_index"]).groupby("regressor_label").size().to_dict()


def counts(summary, subject):
    s = summary[summary["subject"] == subject]
    return {(r.primary_group, r.secondary_group): r.n_trials_restricted for r in s.itertuples()}


class TestExample:
    def test_requested_example(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        _, summary, _ = restrict(tmp_path, build_master(tmp_path, ["001"]))
        assert counts(summary, "001") == {
            ("face", "positive"): 6, ("face", "negative"): 6,
            ("place", "positive"): 6, ("place", "negative"): 6,
            ("item", ""): 12, ("(all)", ""): 36,
        }

    def test_original_counts_reported(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        _, summary, _ = restrict(tmp_path, build_master(tmp_path, ["001"]))
        orig = {(r.primary_group, r.secondary_group): r.n_trials_original for r in summary.itertuples()}
        assert orig[("face", "positive")] == 18 and orig[("item", "")] == 12

    def test_deterministic_and_seed_dependent(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        master = build_master(tmp_path, ["001"])
        a = restrict(tmp_path, master)[2]
        b = restrict(tmp_path, master)[2]
        c = restrict(tmp_path, master, {**CFG, "seed": 8})[2]
        assert a.equals(b)
        assert not a["status"].equals(c["status"])


class TestBalanceMath:
    def test_odd_cap_rounds_down_to_whole_secondary_split(self, tmp_path):
        # item has 7 -> face/place can only take an even 6
        make_subject(tmp_path, "001", [{"face_pos": 10, "face_neg": 10, "place_pos": 10, "place_neg": 10, "item": 7}])
        _, summary, _ = restrict(tmp_path, build_master(tmp_path, ["001"]))
        c = counts(summary, "001")
        assert c[("item", "")] == 6 and c[("face", "positive")] == 3 and c[("place", "negative")] == 3

    def test_secondary_imbalance_is_the_bottleneck(self, tmp_path):
        # 30 face trials but only 4 negative -> 4 pos + 4 neg = 8 face; everyone matches 8
        make_subject(tmp_path, "001", [{"face_pos": 26, "face_neg": 4, "place_pos": 10, "place_neg": 10, "item": 20}])
        _, summary, _ = restrict(tmp_path, build_master(tmp_path, ["001"]))
        c = counts(summary, "001")
        assert c[("face", "positive")] == c[("face", "negative")] == 4
        assert c[("item", "")] == 8

    def test_no_secondary_groups(self, tmp_path):
        make_subject(tmp_path, "001", [{"face_pos": 5, "face_neg": 4, "place_pos": 3, "item": 11}])
        cfg = {"primary_groups": {"face": _q("trial_type", "face_.*"), "place": _q("trial_type", "place_.*"), "item": _q("trial_type", "item")}}
        _, summary, _ = restrict(tmp_path, build_master(tmp_path, ["001"]), cfg)
        c = counts(summary, "001")
        assert c[("face", "")] == c[("place", "")] == c[("item", "")] == 3

    def test_unlabeled_secondary_trial_is_dropped(self, tmp_path):
        make_subject(tmp_path, "001", [{"face_pos": 4, "face_neg": 4, "face_x": 3, "place_pos": 4, "place_neg": 4, "item": 8}])
        _, summary, audit = restrict(tmp_path, build_master(tmp_path, ["001"]))
        assert (audit["status"] == rms.DROPPED_NO_SECONDARY).sum() == 3
        assert counts(summary, "001")[("face", "positive")] == 4

    def test_ungoverned_trials_are_listed_uncapped(self, tmp_path):
        make_subject(tmp_path, "001", [{"face_pos": 4, "face_neg": 4, "place_pos": 4, "place_neg": 4, "item": 8, "other": 5}])
        trial_list, _, _ = restrict(tmp_path, build_master(tmp_path, ["001"]))
        other = trial_list[trial_list["trial_type"] == "other"]
        assert len(other) == 5 and other["n_trs"].isna().all() and other["primary_group"].isna().all()
        assert trial_list["n_trs"].notna().sum() == 24      # the governed ones carry the common TR count

    def test_balanced_across_runs_pooled(self, tmp_path):
        # run 1 holds all the faces, run 2 all the places: balance is per subject, not per run
        make_subject(tmp_path, "001", [{"face_pos": 6, "face_neg": 6, "item": 6}, {"place_pos": 6, "place_neg": 6, "item": 6}])
        _, summary, _ = restrict(tmp_path, build_master(tmp_path, ["001"]))
        c = counts(summary, "001")
        assert c[("face", "positive")] == c[("place", "negative")] == 6 and c[("item", "")] == 12

    def test_spread_runs_keeps_runs_even(self, tmp_path):
        make_subject(tmp_path, "001", [{"face_pos": 8, "face_neg": 8, "place_pos": 8, "place_neg": 8, "item": 4}] * 2)
        cfg = {**CFG, "drop_strategy": "spread_runs"}
        _, _, audit = restrict(tmp_path, build_master(tmp_path, ["001"]), cfg)
        kept = audit[(audit["status"] == rms.KEPT) & (audit["primary_group"] == "face") & (audit["secondary_group"] == "positive")]
        assert kept.groupby("boldfile").size().tolist() == [2, 2]


class TestSubjectExclusion:
    def test_empty_group_excludes_subject_and_reports_why(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        make_subject(tmp_path, "002", [{"face_pos": 5, "face_neg": 5, "place_pos": 5, "place_neg": 5}])  # no items
        trial_list, summary, _ = restrict(tmp_path, build_master(tmp_path, ["001", "002"]))
        assert set(trial_list["subject"]) == {"001"}
        s2 = summary[(summary["subject"] == "002") & (summary["primary_group"] == "(all)")].iloc[0]
        assert s2["status"] == "excluded" and "item" in s2["exclusion_reason"]
        assert s2["n_trials_restricted"] == 0

    def test_missing_secondary_bucket_excludes(self, tmp_path):
        make_subject(tmp_path, "001", [{"face_pos": 5, "place_pos": 5, "place_neg": 5, "item": 5}])
        trial_list, summary, _ = restrict(tmp_path, build_master(tmp_path, ["001"]))
        assert trial_list.empty
        assert "face/negative" in summary["exclusion_reason"].iloc[0]


class TestTRCounts:
    def make_mixed(self, tmp_path):
        # every trial 4 s (2 TR) except one face_pos of 8 s (4 TR) and one item of 6 s (3 TR)
        trials = [("face_pos", 4.0)] * 3 + [("face_pos", 8.0)] + [("face_neg", 4.0)] * 4 \
            + [("place_pos", 4.0)] * 4 + [("place_neg", 4.0)] * 4 + [("item", 4.0)] * 7 + [("item", 6.0)]
        write_run(tmp_path, "001", 1, trials)

    def test_all_governed_trials_share_the_shortest_tr_count(self, tmp_path):
        self.make_mixed(tmp_path)
        trial_list, summary, _ = restrict(tmp_path, build_master(tmp_path, ["001"]))
        assert summary.loc[summary["primary_group"] == "(all)", "trs_per_trial_restricted"].tolist() == [2]
        assert trial_list["n_trs"].tolist() == [2.0] * len(trial_list)

    def test_summary_trs_are_trials_times_trs(self, tmp_path):
        self.make_mixed(tmp_path)
        _, summary, _ = restrict(tmp_path, build_master(tmp_path, ["001"]))
        cells = summary[summary["primary_group"] != "(all)"]
        assert (cells["n_trs_restricted"] == cells["n_trials_restricted"] * 2).all()
        assert cells["n_trs_original"].sum() > cells["n_trs_restricted"].sum()    # 4 + 3 TR trials were longer

    def test_dropping_a_short_trial_raises_the_common_tr_count(self, tmp_path):
        # one 1-TR item among 3-TR item trials: it is the only one dropped (item has 5 > 4), so TRs go 1 -> 3
        trials = [("face_pos", 6.0)] * 2 + [("face_neg", 6.0)] * 2 + [("place_pos", 6.0)] * 2 + [("place_neg", 6.0)] * 2 \
            + [("item", 6.0)] * 4 + [("item", 2.0)]
        write_run(tmp_path, "001", 1, trials)
        master = build_master(tmp_path, ["001"])
        found = {rms.restrict_trials(master, {**CFG, "seed": sd}, 0.0)[1]
                 .query("primary_group == '(all)'")["trs_per_trial_restricted"].iloc[0] for sd in range(40)}
        assert found == {1, 3}      # 3 when the random draw drops the short trial, 1 when it keeps it

    def test_trial_lagged_past_end_of_scan_is_dropped(self, tmp_path):
        trials = [("face_pos", 4.0)] * 2 + [("face_neg", 4.0)] * 2 + [("place_pos", 4.0)] * 2 + [("place_neg", 4.0)] * 2 + [("item", 4.0)] * 4
        write_run(tmp_path, "001", 1, trials)
        master = build_master(tmp_path, ["001"])
        _, summary, _ = restrict(tmp_path, master, lag=0.0)
        assert summary.loc[summary["primary_group"] == "(all)", "trs_per_trial_restricted"].iloc[0] == 2
        # a lag that pushes only the last trial (onset 90 s) past the end of the 400 s scan
        _, _, audit = restrict(tmp_path, master, lag=312.0)
        assert audit["status"].value_counts().get(rms.DROPPED_OUTSIDE_SCAN) == 1


class TestOutputs:
    def test_list_schema_and_lag(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        trial_list, _, _ = restrict(tmp_path, build_master(tmp_path, ["001"]), lag=4.6)
        assert list(trial_list.columns) == ["subject", "boldfile", "trial_index", "onset", "trial_type", "primary_group",
                                            "secondary_group", "n_trs", "hemodynamic_lag"]
        assert (trial_list["hemodynamic_lag"] == 4.6).all() and len(trial_list) == 36

    def test_events_files_and_master_are_untouched(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        master = build_master(tmp_path, ["001"])
        before = {f: open(f).read() for f in master["eventfile"].unique()}
        before_master = master.copy()
        restrict(tmp_path, master)
        assert {f: open(f).read() for f in before} == before
        assert master.equals(before_master)
        assert not (tmp_path / "restricted_events").exists()

    def test_subject_ids_keep_leading_zeros(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        trial_list, _, _ = restrict(tmp_path, build_master(tmp_path, ["001"]))
        assert set(trial_list["subject"]) == {"001"}


class TestFeedsTheWorkflow:
    """The point of the step: training (and so k-fold) selects exactly the listed trials."""

    CONDITIONS = {"face": _q("trial_type", "face_.*"), "place": _q("trial_type", "place_.*"), "item": _q("trial_type", "item")}

    def restricted_labels(self, tmp_path, conditions, lag=4.6, cfg=CFG, subjects=("001",), write=False):
        master = build_master(tmp_path, list(subjects))
        trial_list, _, _ = restrict(tmp_path, master, cfg, lag)
        path = tmp_path / "list.csv"
        trial_list.to_csv(path, index=False)
        table = load_trial_restriction(str(path), lag)
        return master, label_conditions_with_lag(master, conditions, lag, trial_restriction=table)

    def test_training_sees_balanced_equal_length_trials(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        _, labeled = self.restricted_labels(tmp_path, self.CONDITIONS)
        assert n_trials(labeled) == {"face": 12, "place": 12, "item": 12}
        assert labeled.groupby(["boldfile", "trial_index"]).size().nunique() == 1
        assert labeled.groupby("regressor_label").size().nunique() == 1

    def test_without_the_list_training_is_unchanged_and_unbalanced(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        master = build_master(tmp_path, ["001"])
        assert len(set(n_trials(label_conditions_with_lag(master, self.CONDITIONS, 4.6)).values())) > 1

    def test_secondary_balance_visible_to_downstream_queries(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        conds = {"pos": _q("trial_type", ".*_pos"), "neg": _q("trial_type", ".*_neg")}
        _, labeled = self.restricted_labels(tmp_path, conds, lag=0.0)
        assert n_trials(labeled) == {"pos": 12, "neg": 12}

    def test_keeps_the_first_n_trs_and_cuts_from_the_end(self, tmp_path):
        # one 8 s (4 TR) face trial among 4 s ones -> every governed trial limited to 2 TRs
        trials = [("face_pos", 8.0)] + [("face_pos", 4.0)] * 2 + [("face_neg", 4.0)] * 3 + [("place_pos", 4.0)] * 3 \
            + [("place_neg", 4.0)] * 3 + [("item", 4.0)] * 6
        write_run(tmp_path, "001", 1, trials)
        master, labeled = self.restricted_labels(tmp_path, {"face_pos": _q("trial_type", "face_pos")}, lag=0.0)
        unrestricted = label_conditions_with_lag(master, {"face_pos": _q("trial_type", "face_pos")}, 0.0)
        long_onset = 2.0     # the 8 s trial is the first one
        full = unrestricted[unrestricted["onset"] == long_onset]["volume_of_interest_withlag"].tolist()
        kept = labeled[labeled["onset"] == long_onset]["volume_of_interest_withlag"].tolist()
        assert len(full) == 4 and kept == full[:2]
        assert labeled.groupby("trial_index").size().tolist() == [2, 2, 2]

    def test_testing_style_call_without_a_list_is_unaffected(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        master, restricted = self.restricted_labels(tmp_path, self.CONDITIONS)
        assert len(label_conditions_with_lag(master, self.CONDITIONS, 4.6)) > len(restricted)

    def test_unlisted_boldfile_contributes_nothing(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        master = build_master(tmp_path, ["001"])
        trial_list, _, _ = restrict(tmp_path, master, lag=0.0)
        first_run = trial_list["boldfile"].iloc[0]
        trial_list = trial_list[trial_list["boldfile"] != first_run]
        labeled = label_conditions_with_lag(master, self.CONDITIONS, 0.0, trial_restriction=trial_list)
        assert first_run not in set(labeled["boldfile"])

    def test_ungoverned_trials_keep_their_whole_window(self, tmp_path):
        make_subject(tmp_path, "001", [{"face_pos": 4, "face_neg": 4, "place_pos": 4, "place_neg": 4, "item": 8, "other": 5}])
        _, labeled = self.restricted_labels(tmp_path, {"other": _q("trial_type", "other")}, lag=0.0)
        assert n_trials(labeled) == {"other": 5}

    def test_lag_mismatch_is_refused(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        trial_list, _, _ = restrict(tmp_path, build_master(tmp_path, ["001"]), lag=4.6)
        trial_list.to_csv(tmp_path / "list.csv", index=False)
        with pytest.raises(ValueError, match="hemodynamic_lag"):
            load_trial_restriction(str(tmp_path / "list.csv"), 2.0)

    def test_changed_events_file_is_refused(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        master = build_master(tmp_path, ["001"])
        trial_list, _, _ = restrict(tmp_path, master, lag=0.0)
        shifted = trial_list.assign(onset=trial_list["onset"] + 1.0)
        with pytest.raises(ValueError, match="different time"):
            label_conditions_with_lag(master, self.CONDITIONS, 0.0, trial_restriction=shifted)

    def test_missing_file_and_missing_columns_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_trial_restriction(str(tmp_path / "nope.csv"), 0.0)
        (tmp_path / "bad.csv").write_text("subject,boldfile\n1,x\n")
        with pytest.raises(ValueError, match="missing column"):
            load_trial_restriction(str(tmp_path / "bad.csv"), 0.0)


class TestConfigAndCLI:
    def test_validation_catches_problems(self):
        assert rms.validate_restriction_config(CFG) == []
        assert rms.validate_restriction_config({})
        bad = {**CFG, "secondary_applies_to": ["nope"]}
        assert any("nope" in e for e in rms.validate_restriction_config(bad))
        assert rms.validate_restriction_config({**CFG, "drop_strategy": "x"})
        assert rms.validate_restriction_config({**CFG, "typo_key": 1})
        assert rms.validate_restriction_config({"primary_groups": {"a": {"column": "x", "match": "bogus"}}})

    def test_wrapped_config_accepted(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"restriction": CFG}))
        assert rms.load_restriction_config(str(p))["seed"] == 7

    def test_unknown_query_column_gives_readable_error(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        cfg = {"primary_groups": {"face": _q("no_such_column", "x")}}
        with pytest.raises(SystemExit, match="no_such_column"):
            restrict(tmp_path, build_master(tmp_path, ["001"]), cfg)

    def test_cli_writes_all_outputs(self, tmp_path, monkeypatch, capsys):
        make_subject(tmp_path, "001", EXAMPLE)
        master_path = tmp_path / "master.csv"
        build_master(tmp_path, ["001"]).to_csv(master_path, index=False)
        before = master_path.read_text()
        (tmp_path / "r.json").write_text(json.dumps(CFG))
        monkeypatch.setattr("sys.argv", ["x", "--master-spreadsheet", str(master_path), "--restriction-config",
                                         str(tmp_path / "r.json"), "--hemodynamic-lag", "4.6"])
        rms.main()
        for name in ("master_trial_restriction.csv", "master_trial_restriction_summary.csv", "master_trial_restriction_audit.csv"):
            assert (tmp_path / name).is_file()
        assert master_path.read_text() == before
        out = load_trial_restriction(str(tmp_path / "master_trial_restriction.csv"), 4.6)
        assert set(out["subject"]) == {"001"}
        assert "1 of 1 subject" in capsys.readouterr().out

    def test_cli_requires_lag(self, tmp_path, monkeypatch):
        (tmp_path / "r.json").write_text(json.dumps(CFG))
        monkeypatch.setattr("sys.argv", ["x", "--master-spreadsheet", "m.csv", "--restriction-config", str(tmp_path / "r.json")])
        with pytest.raises(SystemExit, match="hemodynamic"):
            rms.main()


class TestUngovernedWarning:
    def test_flags_condition_trials_outside_every_primary_group(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        master = build_master(tmp_path, ["001"])
        # primary group "face" only governs run 1, but the condition selects faces from both runs
        cfg = {"primary_groups": {"face": {"and": [_q("trial_type", "face_.*"), _q("run", "1")]}}}
        trials = rms.collect_trials(master, cfg, 0.0)
        found = rms.find_ungoverned_condition_trials(master, trials, {"training": {"conditions": {"face": _q("trial_type", "face_.*")}}}, 0.0)
        assert found == [("training", "face", 18, 36)]

    def test_silent_when_primary_groups_cover_the_condition(self, tmp_path):
        make_subject(tmp_path, "001", EXAMPLE)
        master = build_master(tmp_path, ["001"])
        trials = rms.collect_trials(master, CFG, 0.0)
        mc = {"training": {"conditions": {"face": _q("trial_type", "face_.*"), "item": _q("trial_type", "item")}}}
        assert rms.find_ungoverned_condition_trials(master, trials, mc, 0.0) == []
