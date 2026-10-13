"""trial_order_analysis: transition tables, balance tests and the baseline-window
check, on small synthetic acquisitions in the group decoding-table format."""

import numpy as np
import pandas as pd
import pytest

from _interactive_notebooks import trial_order_analysis as toa


def acquisition(subject, boldfile, trials, task="WMpos", run=1, session="A1", baseline_ev=None, fixation_trs=None):
    """Decoding-table rows for one acquisition. `trials` is a list of trial_type
    strings (e.g. "maintain_face"); None makes a malformed trial (view_nan, no
    operation event). Each trial = 3 view TRs, 2 operation TRs, 1 fixation TR.
    Trial 0 is the 3-TR pre-trial fixation. fixation_trs, if given, is the fixation
    length (TRs) after each trial (default 1). baseline_ev, if given, is a function
    (trial position, type) -> evidence for the trial's own category in its first TRs."""
    rows, vol = [], 0
    for w in range(3):
        rows.append(dict(trial_index=0, window_index=w, volume_of_interest=vol, trial_type="BaselineFixation", ev=0.5)); vol += 1
    for i, tt in enumerate(trials, start=1):
        category = tt.split("_")[1] if tt else "nan"
        fix = fixation_trs[i - 1] if fixation_trs else 1
        seq = [f"view_{category}"] * 3 + ([tt] * 2 if tt else ["nan_nan"] * 2) + ["trial_fixation"] * fix
        for w, t in enumerate(seq):
            ev = baseline_ev(i, tt) if (baseline_ev and w < 3) else 0.5
            rows.append(dict(trial_index=i, window_index=w, volume_of_interest=vol, trial_type=t, ev=ev)); vol += 1
    df = pd.DataFrame(rows)
    df["subject"], df["boldfile"], df["task"], df["run"], df["session"] = subject, boldfile, task, run, session
    # the evidence columns: both categories carry `ev`, so self-evidence is `ev` whichever category the trial has
    df["evidence_face"] = df["ev"]; df["evidence_place"] = df["ev"]
    return df.drop(columns="ev")


def decoding(*acqs):
    return pd.concat(acqs, ignore_index=True)


ORDER = ["maintain_face", "clear_place", "switch_face", "suppress_place"]


class TestTrialTable:
    def test_one_row_per_real_trial_with_type_and_previous(self):
        t = toa.build_trial_table(decoding(acquisition("1", "a", ORDER)))
        assert len(t) == 4                                            # trial 0 (pre-trial fixation) is not a trial
        assert t["trial_type"].tolist() == ORDER
        assert t["previous_trial"].tolist() == [toa.START] + ORDER[:-1]
        assert t["operation"].tolist() == ["maintain", "clear", "switch", "suppress"]
        assert t["category"].tolist() == ["face", "place", "face", "place"]
        assert set(t["valence"]) == {"pos"}

    def test_first_trial_of_each_acquisition_is_preceded_by_start(self):
        t = toa.build_trial_table(decoding(acquisition("1", "a", ORDER), acquisition("1", "b", ORDER[::-1], run=2)))
        firsts = t.groupby("boldfile").head(1)
        assert (firsts["previous_trial"] == toa.START).all() and len(firsts) == 2

    def test_previous_trial_never_crosses_an_acquisition(self):
        t = toa.build_trial_table(decoding(acquisition("1", "a", ["maintain_face", "clear_place"]),
                                           acquisition("1", "b", ["switch_face", "suppress_place"], run=2)))
        assert t.loc[t["trial_type"] == "switch_face", "previous_trial"].iloc[0] == toa.START

    def test_malformed_trial_has_no_type_and_is_left_out_of_the_transitions(self):
        t = toa.build_trial_table(decoding(acquisition("1", "a", ORDER + [None])))
        assert len(t) == 5 and t["trial_type"].isna().sum() == 1
        usable = toa.usable_transitions(t)
        assert len(usable) == 4 and usable["trial_type"].notna().all()

    def test_trial_order_follows_time_not_label(self):
        df = acquisition("1", "a", ORDER)
        shuffled = df.sample(frac=1, random_state=0)
        t = toa.build_trial_table(shuffled)
        assert t["trial_type"].tolist() == ORDER


class TestLabelsAndIti:
    def test_labels_carry_the_valence_and_start_is_unchanged(self):
        t = toa.build_trial_table(decoding(acquisition("1", "a", ORDER, task="WMneg")))
        assert t["trial_label"].tolist() == [f"{x}_neg" for x in ORDER]
        assert t["previous_label"].tolist() == [toa.START] + [f"{x}_neg" for x in ORDER[:-1]]

    def test_iti_is_the_gap_between_the_previous_operation_and_this_view(self):
        t = toa.build_trial_table(decoding(acquisition("1", "a", ORDER, fixation_trs=[2, 5, 3, 4])))
        # trial 1 follows the 3-TR lead-in; trials 2-4 follow the fixation of the trial before them
        assert t["iti_trs"].tolist() == [3, 2, 5, 3]

    def test_iti_never_crosses_an_acquisition(self):
        d = decoding(acquisition("1", "a", ORDER, fixation_trs=[9, 9, 9, 9]), acquisition("1", "b", ORDER, run=2, fixation_trs=[1, 1, 1, 1]))
        t = toa.build_trial_table(d)
        first_b = t[(t["boldfile"] == "b")].iloc[0]
        assert first_b["iti_trs"] == 3                               # lead-in of acquisition b, not related to a's last fixation

    def test_unlabelled_gap_volumes_count_toward_the_iti(self):
        d = acquisition("1", "a", ORDER, fixation_trs=[2, 2, 2, 2])
        # remove one fixation volume's label of trial 1 -> an unlabelled gap volume still lies between operation and next view
        gap_vol = d[(d["trial_index"] == 1) & (d["window_index"] == 6)]["volume_of_interest"].iloc[0]
        d.loc[d["volume_of_interest"] == gap_vol, "trial_type"] = np.nan
        t = toa.build_trial_table(d)
        assert t["iti_trs"].tolist()[1] == 2

    def test_iti_after_a_typeless_trial_is_unknown(self):
        t = toa.build_trial_table(decoding(acquisition("1", "a", ["maintain_face", None, "clear_place"])))
        assert np.isnan(t["iti_trs"].iloc[2])


class TestCells:
    def test_cell_format_total_then_each_iti_length(self):
        assert toa.iti_cell([8, 8, 9]) == "3 (8TR x2, 9TR x1)"
        assert toa.iti_cell([9, 5, 5, 5]) == "4 (5TR x3, 9TR x1)"         # shortest first
        assert toa.iti_cell([]) == "0"
        assert toa.iti_cell([7, np.nan]) == "2 (7TR x1, NA x1)"

    def test_cells_table_total_column_and_zero_cells(self):
        d = decoding(acquisition("1", "a", ["maintain_face", "clear_place", "maintain_face", "clear_place"], fixation_trs=[2, 3, 2, 3]))
        t = toa.usable_transitions(toa.build_trial_table(d))
        labels = ["clear_place_pos", "maintain_face_pos"]
        c = toa.transition_cells(t, previous_order=[toa.START] + labels, current_order=labels)
        assert c.loc["maintain_face_pos", "clear_place_pos"] == "2 (2TR x1, 3TR x1)" or c.loc["maintain_face_pos", "clear_place_pos"].startswith("2 (")
        assert c.loc["clear_place_pos", "clear_place_pos"] == "0"
        assert c.loc["maintain_face_pos", "total"] == "2"
        assert c.loc[toa.START, "maintain_face_pos"] == "1 (3TR x1)"      # the lead-in

    def test_cell_totals_equal_the_numeric_counts(self):
        d = decoding(acquisition("1", "a", ORDER * 3, fixation_trs=[1, 2, 3, 4] * 3))
        t = toa.usable_transitions(toa.build_trial_table(d))
        labels = sorted(t["trial_label"].unique()); order = [toa.START] + labels
        numeric = toa.transition_counts(t, "previous_label", "trial_label", previous_order=order, current_order=labels, margins=False)
        text = toa.transition_cells(t, previous_order=order, current_order=labels).drop(columns="total")
        parsed = text.apply(lambda col: col.str.extract(r"^(\d+)")[0].astype(int))
        assert (parsed.to_numpy() == numeric.to_numpy()).all()


class TestIti:
    def test_distribution_counts_between_trial_itis_and_reports_lead_in_separately(self):
        d = decoding(acquisition("1", "a", ORDER, fixation_trs=[2, 5, 3, 4]))
        dist, summ = toa.iti_summary(toa.build_trial_table(d), tr_seconds=0.5)
        between = dist[dist["kind"] == "iti"].set_index("iti_trs")["n"].to_dict()
        assert between == {2: 1, 5: 1, 3: 1}                       # the last fixation (4 TR) precedes no trial
        assert dist[dist["kind"] == "lead-in"]["iti_trs"].tolist() == [3]
        assert dist["iti_seconds"].tolist()[0] == pytest.approx(dist["iti_trs"].tolist()[0] * 0.5)
        assert set(summ["kind"]) == {"iti", "lead-in"}

    def test_dependence_test_is_nan_when_the_iti_is_constant(self):
        d = decoding(acquisition("1", "a", ORDER * 3))
        out = toa.iti_dependence_tests(toa.build_trial_table(d))
        assert out["p"].isna().all()

    def test_dependence_test_detects_an_iti_tied_to_the_trial_type(self):
        order = ["maintain_face", "clear_place", "switch_face", "suppress_place"] * 6
        fix = [{"maintain_face": 2, "clear_place": 6, "switch_face": 4, "suppress_place": 8}[o] for o in order]
        d = decoding(*[acquisition(str(s), f"b{s}", order, fixation_trs=fix) for s in range(3)])
        out = toa.iti_dependence_tests(toa.build_trial_table(d))
        assert (out.loc[out["valence"] == "pos", "p"] < 1e-6).all()


class TestCounts:
    def test_counts_rows_are_previous_and_columns_current(self):
        t = toa.build_trial_table(decoding(acquisition("1", "a", ORDER)))
        c = toa.transition_counts(toa.usable_transitions(t), previous_order=[toa.START] + ORDER, current_order=ORDER, margins=False)
        assert c.loc["maintain_face", "clear_place"] == 1
        assert c.loc[toa.START, "maintain_face"] == 1
        assert c.loc["clear_place", "maintain_face"] == 0
        assert c.to_numpy().sum() == 4

    def test_axes_can_be_forced_so_tables_line_up(self):
        t = toa.build_trial_table(decoding(acquisition("1", "a", ["maintain_face", "clear_place"])))
        c = toa.transition_counts(t, previous_order=[toa.START] + ORDER, current_order=ORDER)
        assert list(c.index) == [toa.START] + ORDER and list(c.columns) == ORDER + ["total"]

    def test_row_percent_sums_to_100_where_there_are_trials(self):
        t = toa.build_trial_table(decoding(acquisition("1", "a", ORDER * 3)))
        c = toa.transition_counts(toa.usable_transitions(t), previous_order=[toa.START] + ORDER, current_order=ORDER)
        pct = toa.row_percent(c)
        assert np.allclose(pct.sum(axis=1).dropna(), 100)
        assert pct.loc[toa.START].isna().sum() == 0

    def test_group_table_is_the_sum_of_the_subject_tables(self):
        d = decoding(acquisition("1", "a", ORDER), acquisition("2", "b", ORDER[::-1], run=2))
        t = toa.usable_transitions(toa.build_trial_table(d))
        all_ = toa.transition_counts(t, previous_order=[toa.START] + ORDER, current_order=ORDER, margins=False)
        parts = sum(toa.transition_counts(g, previous_order=[toa.START] + ORDER, current_order=ORDER, margins=False)
                    for _, g in t.groupby("subject"))
        assert all_.equals(parts)


class TestIndependenceTest:
    TYPES = ["a", "b", "c"]

    def _counts(self, matrix):
        return pd.DataFrame(matrix, index=self.TYPES, columns=self.TYPES)

    def test_perfectly_counterbalanced_gives_zero_chi_square(self):
        r = toa.independence_test(self._counts(np.full((3, 3), 7)))
        assert r["chi2"] == pytest.approx(0.0) and r["p"] == pytest.approx(1.0) and r["cramers_v"] == pytest.approx(0.0)
        assert r["min_cell"] == r["max_cell"] == 7 and r["n_transitions"] == 63

    def test_strong_dependence_is_detected(self):
        r = toa.independence_test(self._counts(np.eye(3, dtype=int) * 30 + 1))
        assert r["p"] < 1e-10 and r["cramers_v"] > 0.8

    def test_start_row_is_not_a_transition(self):
        c = self._counts(np.full((3, 3), 7))
        c.loc[toa.START] = [100, 0, 0]                           # an unbalanced <start> row must be ignored
        assert toa.independence_test(c)["chi2"] == pytest.approx(0.0)

    def test_degenerate_table_gives_nan_not_an_error(self):
        r = toa.independence_test(self._counts(np.zeros((3, 3), dtype=int)))
        assert np.isnan(r["chi2"]) and r["n_transitions"] == 0


class TestSequences:
    def test_same_order_gets_the_same_id_and_different_orders_differ(self):
        d = decoding(acquisition("1", "a", ORDER), acquisition("2", "b", ORDER),
                     acquisition("1", "c", ORDER[::-1], run=2))
        s = toa.sequence_table(toa.build_trial_table(d)).set_index("boldfile")
        assert s.loc["a", "sequence_id"] == s.loc["b", "sequence_id"] != s.loc["c", "sequence_id"]

    def test_an_order_missing_its_last_typed_trial_joins_the_full_order(self):
        d = decoding(acquisition("1", "full", ORDER + ["switch_place"]), acquisition("2", "short", ORDER + [None]))
        s = toa.sequence_table(toa.build_trial_table(d)).set_index("boldfile")
        assert s.loc["full", "sequence_id"] == s.loc["short", "sequence_id"]
        assert s.loc["short", "n_typed_trials"] == 4 and s.loc["short", "n_trials"] == 5
        assert not s["typeless_trial_not_last"].any()

    def test_a_typeless_trial_in_the_middle_is_flagged(self):
        s = toa.sequence_table(toa.build_trial_table(decoding(acquisition("1", "a", ["maintain_face", None, "clear_place"]))))
        assert bool(s["typeless_trial_not_last"].iloc[0])

    def test_composition_counts_each_type_in_the_order(self):
        d = decoding(acquisition("1", "a", ["maintain_face", "maintain_face", "clear_place"]))
        t = toa.build_trial_table(d)
        comp = toa.order_composition(toa.sequence_table(t), sorted(t["trial_type"].dropna().unique()))
        assert comp.loc[0, "maintain_face"] == 2 and comp.loc[0, "clear_place"] == 1 and comp.loc[0, "n_typed_trials"] == 3


class TestSameCategory:
    def test_percentage_of_trials_that_follow_the_same_category(self):
        # clear_face follows a face trial once and a place trial once
        order = ["maintain_face", "clear_face", "maintain_place", "clear_face"]
        r = toa.same_category_rate(toa.build_trial_table(decoding(acquisition("1", "a", order))))
        assert r.loc["clear_face", "n"] == 2 and r.loc["clear_face", "percent_same_category"] == pytest.approx(50.0)
        assert r.loc["maintain_place", "percent_same_category"] == 0.0     # follows a face trial


class TestBaselineWindow:
    def test_trial_baseline_is_the_mean_of_the_first_windows_for_its_own_category(self):
        d = acquisition("1", "a", ORDER, baseline_ev=lambda i, tt: 0.1 * i)
        t = toa.baseline_by_trial(d, toa.build_trial_table(d), baseline_windows=(0, 1, 2))
        assert np.allclose(t["baseline_evidence"], [0.1, 0.2, 0.3, 0.4])

    def test_only_the_requested_windows_are_used(self):
        d = acquisition("1", "a", ORDER, baseline_ev=lambda i, tt: 0.5)
        d.loc[d["window_index"] == 2, ["evidence_face", "evidence_place"]] = 0.8
        t1 = toa.baseline_by_trial(d, toa.build_trial_table(d), baseline_windows=(0, 1))
        t2 = toa.baseline_by_trial(d, toa.build_trial_table(d), baseline_windows=(0, 1, 2))
        assert np.allclose(t1["baseline_evidence"], 0.5) and np.allclose(t2["baseline_evidence"], 0.6)

    def test_anova_detects_a_level_effect_and_ignores_subject_offsets(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(20, 1)) * 5                       # big subject offsets
        flat = pd.DataFrame(base + rng.normal(size=(20, 4)) * 0.01)
        assert toa.repeated_measures_anova(flat)["p"] > 0.01
        effect = flat.copy(); effect[0] += 1.0
        r = toa.repeated_measures_anova(effect)
        assert r["p"] < 1e-10 and r["df1"] == 3 and r["df2"] == 57 and 0 < r["partial_eta_sq"] <= 1
        shifted = effect + base                                   # subject offsets must not change F
        assert toa.repeated_measures_anova(shifted)["F"] == pytest.approx(r["F"], rel=1e-6)

    def test_same_category_predecessor_raises_baseline_in_a_planted_effect(self):
        order = ["maintain_face", "clear_face", "maintain_place", "clear_face", "switch_face", "suppress_place"] * 2
        frames = []
        for s in range(6):
            frames.append(acquisition(str(s), f"b{s}", order,
                                      baseline_ev=lambda i, tt, o=order: 0.8 if i > 1 and o[i - 2].split("_")[1] == tt.split("_")[1] else 0.3))
        d = decoding(*frames)
        trials = toa.baseline_by_trial(d, toa.build_trial_table(d))
        table, tests = toa.baseline_summary(trials)
        diff = tests.loc[tests["test"].str.startswith("same-category"), "mean_difference"].iloc[0]
        assert diff == pytest.approx(0.5)

    def test_no_same_category_predecessor_anywhere_gives_nan_not_an_error(self):
        d = decoding(acquisition("1", "a", ORDER * 2, baseline_ev=lambda i, tt: 0.5))      # categories strictly alternate
        _, tests = toa.baseline_summary(toa.baseline_by_trial(d, toa.build_trial_table(d)))
        row = tests[tests["test"].str.startswith("same-category")].iloc[0]
        assert row["n_subjects"] == 0 and np.isnan(row["p"])

    def test_by_current_type_reports_share_and_mean(self):
        order = ["maintain_face", "clear_face", "maintain_place", "clear_face"]
        d = decoding(acquisition("1", "a", order, baseline_ev=lambda i, tt: 0.5))
        out = toa.baseline_by_current_type(toa.baseline_by_trial(d, toa.build_trial_table(d)))
        assert out.loc["clear_face", "n_trials"] == 2 and out.loc["clear_face", "percent_same_category"] == pytest.approx(50.0)
        assert out.loc["clear_face", "mean_baseline_evidence"] == pytest.approx(0.5)


class TestValenceTables:
    def _two_valences(self):
        return decoding(*[acquisition(str(subj), f"{subj}-{r}", ORDER * 3, run=r, task=("WMpos" if r % 2 else "WMneg"),
                                      fixation_trs=[3, 4, 5, 6] * 3)
                          for subj in range(2) for r in (1, 2)])

    def test_pos_and_neg_tables_never_mix_valences(self):
        res = toa.analyse(self._two_valences())
        for valence, other in (("pos", "neg"), ("neg", "pos")):
            counts = res[f"group_counts_{valence}"]
            assert all(x == toa.START or x.endswith(f"_{valence}") for x in counts.index)
            assert all(x == "total" or x.endswith(f"_{valence}") for x in counts.columns)
            assert not any(other in str(x) for x in list(counts.index) + list(counts.columns))

    def test_each_valence_table_counts_only_its_own_acquisitions(self):
        res = toa.analyse(self._two_valences())
        assert res["group_counts_pos"]["total"].sum() == res["group_counts_neg"]["total"].sum() == 2 * 12   # 2 subjects x 12 trials
        assert res["group_counts_pos"].loc[toa.START, "total"] == 2           # one first trial per acquisition per subject

    def test_there_is_no_pooled_table(self):
        res = toa.analyse(self._two_valences())
        assert "group_counts" not in res and "group_counts_pos" in res and "group_cells_neg" in res


class TestEndToEnd:
    def test_analyse_and_save_write_every_table(self, tmp_path):
        d = decoding(*[acquisition(str(s), f"{s}-{r}", ORDER * 3, run=r, task=("WMpos" if r % 2 else "WMneg"),
                                   fixation_trs=[3, 4, 5, 6] * 3)
                       for s in range(3) for r in (1, 2)])
        res = toa.analyse(d)
        for stale in toa.STALE_FILES:                               # old pooled files must not survive a re-run
            (tmp_path / stale).write_text("old")
        files = toa.save(res, str(tmp_path))
        for valence in ("pos", "neg"):
            for name in (f"transitions_group_{valence}_with_iti.csv", f"transitions_group_{valence}_counts.csv",
                         f"transitions_by_subject_{valence}_with_iti.csv", f"transitions_by_subject_{valence}_counts.csv",
                         f"transitions_group_{valence}_heatmap.png"):
                assert name in files and (tmp_path / name).is_file()
        assert not any((tmp_path / stale).exists() for stale in toa.STALE_FILES)
        assert res["summary"].loc["subjects", "value"] == 3 and res["summary"].loc["distinct_trial_orders", "value"] == 1   # the same list is used in both valences here
        cells = pd.read_csv(tmp_path / "transitions_group_pos_with_iti.csv").set_index("previous_label")
        assert cells.loc[toa.START, "total"] == 3                  # one first trial per pos acquisition (3 subjects)
        assert cells.loc[toa.START, "maintain_face_pos"] == "3 (3TR x3)"
        wide = pd.read_csv(tmp_path / "transitions_by_subject_neg_with_iti.csv", dtype={"subject": str})
        assert wide["subject"].nunique() == 3 and set(wide["previous_trial"]) == {toa.START, *[f"{x}_neg" for x in ORDER]}
        long = pd.read_csv(tmp_path / "transitions_by_subject_long.csv", dtype={"subject": str})
        assert set(long["valence"]) == {"pos", "neg"} and {"count", "iti_counts"} <= set(long.columns)


# ---------------------------------------------------------------------------
# baseline bias, restricting to different-category predecessors, dropped trials
# ---------------------------------------------------------------------------

SAME_SHARE = {("maintain", "face"): 0.25, ("maintain", "place"): 0.50, ("clear", "face"): 0.75, ("clear", "place"): 0.50}


def fake_trial_table(n_subjects=10, same_effect=0.30, iti_slope=0.05, n_per_label=40, seed=0, same_share=None):
    """A trial table with a planted baseline: base 0.4, +same_effect after a same-category
    predecessor, +iti_slope per TR of ITI after a different-category predecessor."""
    share = same_share or SAME_SHARE
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for valence in ("pos", "neg"):
            for (op, cat), p_same in share.items():
                n_same = round(p_same * n_per_label)
                for k in range(n_per_label):
                    same = k < n_same
                    prev_cat = cat if same else ("place" if cat == "face" else "face")
                    iti = int(rng.integers(5, 10))
                    base = 0.4 + (same_effect if same else iti_slope * (iti - 7)) + rng.normal(0, 0.01)
                    rows.append(dict(subject=f"s{s}", boldfile=f"s{s}-{valence}", valence=valence, trial_type=f"{op}_{cat}",
                                     trial_label=f"{op}_{cat}_{valence}", operation=op, category=cat,
                                     previous_trial=f"maintain_{prev_cat}", previous_label=f"maintain_{prev_cat}_{valence}",
                                     iti_trs=iti, baseline_evidence=base))
                # one acquisition-opening trial per label: no predecessor, must never enter the transition sets
                rows.append(dict(subject=f"s{s}", boldfile=f"s{s}-{valence}", valence=valence, trial_type=f"{op}_{cat}",
                                 trial_label=f"{op}_{cat}_{valence}", operation=op, category=cat, previous_trial=toa.START,
                                 previous_label=toa.START, iti_trs=59, baseline_evidence=5.0))
    return pd.DataFrame(rows)


class TestMidTrialPanel:
    def _results(self):
        d = decoding(*[acquisition(str(s), f"{s}-{r}", ORDER * 3, run=r, task=("WMpos" if r % 2 else "WMneg"), fixation_trs=[9] * 12)
                       for s in range(3) for r in (1, 2)])
        # make the evidence in TRs 11-13 (window_index 10-12) distinguishable from the baseline TRs and the rest
        d.loc[d["window_index"].isin([10, 11, 12]), ["evidence_face", "evidence_place"]] = 0.9
        return toa.analyse(d)

    def test_trial_table_has_the_tr_11_13_self_evidence(self):
        twb = self._results()["trials_with_baseline"]
        typed = twb[twb["trial_type"].notna()]
        assert typed[toa.MID_TRIAL_COLUMN].eq(0.9).all()
        assert not typed["baseline_evidence"].eq(0.9).any()           # the baseline window is untouched

    def test_plot_has_three_panels_with_the_mid_trial_one_in_the_middle(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        res = self._results()
        fig = toa.plot_carryover(res["trials_with_baseline"], res["bias_by_trial_type"])
        assert len(fig.axes) == 3 and "TRs 11-13" in fig.axes[1].get_title() and "TRs 1-3" in fig.axes[0].get_title()
        assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()      # same scale, so the two panels can be compared
        plt.close(fig)

    def test_plot_without_the_mid_trial_column_keeps_two_panels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        res = self._results()
        fig = toa.plot_carryover(res["trials_with_baseline"].drop(columns=toa.MID_TRIAL_COLUMN), res["bias_by_trial_type"])
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_per_subject_table_has_both_windows(self):
        t = toa.baseline_by_subject_predecessor(self._results()["trials_with_baseline"])
        assert {"tr1_3_same_minus_different", "tr11_13_after_same_category", "tr11_13_after_different_category"} <= set(t.columns)
        assert t["tr11_13_after_same_category"].dropna().eq(0.9).all()


class TestBaselineBias:
    def test_same_category_part_equals_the_share_times_the_effect(self):
        table, _, _, _ = toa.baseline_bias(fake_trial_table())
        t = table.set_index("trial_label")
        for (op, cat), share in SAME_SHARE.items():
            assert t.loc[f"{op}_{cat}_pos", "bias_same_category"] == pytest.approx(0.30 * share, abs=0.01)
            assert t.loc[f"{op}_{cat}_pos", "percent_same_category_predecessor"] == pytest.approx(share * 100)

    def test_first_trials_never_enter_the_transition_sets(self):
        table, _, _, _ = toa.baseline_bias(fake_trial_table())
        t = table.set_index("trial_label")
        assert (t["baseline_transitions"] < 1.0).all()                      # the 5.0 first-trial baselines are not in it
        assert (t["baseline_notebook"] > t["baseline_transitions"]).all()   # the notebook set does include them

    def test_iti_slope_is_recovered_within_subject_and_type(self):
        _, _, tests, _ = toa.baseline_bias(fake_trial_table())
        row = tests[tests["trial_set"].str.startswith("ITI slope, different")].set_index("valence")
        assert row.loc["pos", "type_mean_range"] == pytest.approx(0.05, abs=0.005)       # slope lives in this column
        assert (row["p"] < 1e-6).all()
        same = tests[tests["trial_set"].str.startswith("ITI slope, same")].set_index("valence")
        assert (same["type_mean_range"].abs() < 0.01).all()                              # no ITI effect after a same-category trial

    def test_restricting_flattens_a_difference_that_came_from_the_predecessors(self):
        _, _, tests, _ = toa.baseline_bias(fake_trial_table())
        t = tests[(tests["valence"] == "pos")].set_index("trial_set")
        assert t.loc["notebook", "type_mean_range"] > 0.12 > 0.05 > t.loc["different", "type_mean_range"]
        assert t.loc["different", "partial_eta_sq"] < t.loc["notebook", "partial_eta_sq"]
        assert t.loc["notebook", "p"] < 1e-6

    def test_pairwise_matrix_is_antisymmetric_with_a_zero_diagonal(self):
        table, pairwise, _, _ = toa.baseline_bias(fake_trial_table())
        m = pairwise["pos"].to_numpy()
        assert np.allclose(m, -m.T) and np.allclose(np.diag(m), 0)
        assert m.shape == (4, 4)
        bt = table.set_index("trial_label")["bias_total"]
        assert pairwise["pos"].loc["clear_face_pos", "maintain_face_pos"] == pytest.approx(bt["clear_face_pos"] - bt["maintain_face_pos"])

    def test_no_same_category_effect_means_no_bias(self):
        table, _, _, _ = toa.baseline_bias(fake_trial_table(same_effect=0.0, iti_slope=0.0))
        assert table["bias_same_category"].abs().max() < 0.01 and table["bias_iti"].abs().max() < 0.01


class TestContrastBias:
    def test_contrasts_match_the_notebooks_and_signs_are_flipped(self):
        table, _, _, _ = toa.baseline_bias(fake_trial_table())
        c = toa.contrast_bias(table)
        assert set(c["contrast_type"]) == {"operation vs maintain", "pos vs neg"}
        assert len(c[c["contrast_type"] == "operation vs maintain"]) == 4         # clear - maintain: 2 categories x 2 valences
        assert len(c[c["contrast_type"] == "pos vs neg"]) == 4                    # 2 operations x 2 categories
        assert np.allclose(c["implied_shift_of_corrected_difference"], -c["bias_total"])
        row = c[c["contrast"] == "clear - maintain (face, pos)"].iloc[0]
        t = table.set_index("trial_label")
        assert row["bias_total"] == pytest.approx(t.at["clear_face_pos", "bias_total"] - t.at["maintain_face_pos", "bias_total"])
        assert row["bias_total"] == pytest.approx(0.30 * (0.75 - 0.25), abs=0.02)


# acquisitions where two trials follow a trial of their own category
SAME_ORDER = ["maintain_face", "clear_face", "switch_place", "suppress_place", "clear_face"]   # clear_face, suppress_place follow same category


class TestDropped:
    def _trials(self):
        d = decoding(acquisition("1", "a", SAME_ORDER, task="WMpos"), acquisition("1", "b", SAME_ORDER, run=2, task="WMneg"),
                     acquisition("2", "c", SAME_ORDER, task="WMpos"))
        return toa.build_trial_table(d)

    def test_counts_per_subject_and_valence(self):
        per, _ = toa.dropped_same_category(self._trials())
        r = per.set_index(["subject", "valence"])
        assert r.loc[("1", "pos"), "n_same_category_dropped"] == 2 and r.loc[("1", "neg"), "n_same_category_dropped"] == 2
        assert r.loc[("1", "all"), "n_same_category_dropped"] == 4 and r.loc[("1", "all"), "n_trials"] == 10
        assert r.loc[("2", "all"), "n_same_category_dropped"] == 2
        assert r.loc[("1", "pos"), "n_first_trials"] == 1

    def test_first_trial_of_an_acquisition_is_kept(self):
        per, kept = toa.dropped_same_category(self._trials())
        assert kept.loc["1", "maintain_face_pos"] == 1                   # the opening trial, no predecessor
        r = per.set_index(["subject", "valence"]).loc[("1", "pos")]
        assert r["n_kept"] == r["n_trials"] - r["n_same_category_dropped"] == 3

    def test_percent_and_kept_table(self):
        per, kept = toa.dropped_same_category(self._trials())
        r = per.set_index(["subject", "valence"]).loc[("1", "pos")]
        assert r["percent_dropped"] == pytest.approx(40.0)
        assert kept.loc["1", "clear_face_pos"] == 1                      # the 2nd trial follows a face trial (dropped); the last follows a place trial (kept)
        assert kept.loc["1", "min_per_trial_type"] == kept.loc["1"].drop("min_per_trial_type").min()

    def test_dropped_by_trial_type_counts_and_percents(self):
        t = toa.dropped_by_trial_type(self._trials())
        assert t["n_dropped"].sum() == 6 and t["n_trials"].sum() == 15
        r = t.set_index("trial_label").loc["clear_face_pos"]          # 2 clear_face per acquisition x 2 pos acquisitions; the first follows a face trial, the second a place trial
        assert r["n_trials"] == 4 and r["n_dropped"] == 2 and r["percent_dropped"] == pytest.approx(50.0)

    def test_dropped_plot_is_a_single_panel(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = toa.plot_dropped(toa.dropped_by_trial_type(self._trials()))
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_a_trial_after_an_untyped_trial_is_not_counted_as_dropped(self):
        d = decoding(acquisition("1", "a", ["maintain_face", None, "clear_face"]))
        per, _ = toa.dropped_same_category(toa.build_trial_table(d))
        assert per.set_index(["subject", "valence"]).loc[("1", "pos"), "n_same_category_dropped"] == 0


class TestNewOutputs:
    def test_analyse_save_writes_the_bias_files_and_plots(self, tmp_path):
        d = decoding(*[acquisition(str(s), f"{s}-{r}", SAME_ORDER * 3, run=r, task=("WMpos" if r % 2 else "WMneg"),
                                   fixation_trs=[3, 4, 5, 6, 4] * 3, baseline_ev=lambda i, tt: 0.3 + 0.01 * i)
                       for s in range(4) for r in (1, 2)])
        res = toa.analyse(d)
        files = toa.save(res, str(tmp_path))
        for name in ("baseline_bias_by_trial_type.csv", "baseline_bias_tests.csv", "baseline_bias_in_notebook_contrasts.csv",
                     "dropped_same_category_per_subject.csv", "dropped_same_category_by_trial_type.csv", "kept_trials_per_subject_trial_type.csv", "baseline_confound_plots.pdf",
                     "baseline_bias_pairwise_pos.csv", "baseline_bias_pairwise_neg.csv", "baseline_by_subject_and_predecessor.csv", "plot_1_baseline_vs_same_category_predecessor.png",
                     "plot_6_bias_in_the_notebook_contrasts.png"):
            assert name in files and (tmp_path / name).is_file(), name
        assert (tmp_path / "baseline_confound_plots.pdf").stat().st_size > 5000


class TestPlotsOnThinData:
    def test_contrast_plot_ignores_missing_contrasts(self):
        import matplotlib
        matplotlib.use("Agg")
        table, _, _, _ = toa.baseline_bias(fake_trial_table(same_share={("maintain", "face"): 0.5, ("clear", "face"): 0.5}))
        fig = toa.plot_contrast_bias(toa.contrast_bias(table))        # no place trials at all: those contrasts do not exist
        assert len(fig.axes) == 2
        import matplotlib.pyplot as plt; plt.close(fig)


# ---------------------------------------------------------------------------
# self / off-category / item evidence at the baseline, by ITI
# ---------------------------------------------------------------------------

def fake_iti_table(n_subjects=8, per_cell=15):
    """Baseline evidence planted as: after a DIFFERENT-category predecessor the previous
    (off) category fades and the current category recovers with the ITI; after a SAME
    category predecessor the current category stays high and the other category recovers."""
    rows = []
    for s in range(n_subjects):
        for same in (False, True):
            for iti in range(5, 11):
                for k in range(per_cell):
                    cat = "face" if k % 2 == 0 else "place"
                    prev = cat if same else ("place" if cat == "face" else "face")
                    x = iti - 7
                    if same:
                        self_, off, item = 0.65, 0.30 + 0.04 * x, 0.55 + 0.06 * x
                    else:
                        self_, off, item = 0.40 + 0.05 * x, 0.62 - 0.03 * x, 0.55 + 0.06 * x
                    rows.append(dict(subject=f"s{s}", boldfile=f"s{s}", valence="pos", trial_type=f"maintain_{cat}", trial_label=f"maintain_{cat}_pos",
                                     category=cat, operation="maintain", previous_trial=f"maintain_{prev}", iti_trs=iti, baseline_evidence=self_,
                                     baseline_off_category_evidence=off, baseline_item_evidence=item))
        rows.append(dict(subject=f"s{s}", boldfile=f"s{s}", valence="pos", trial_type="maintain_place", trial_label="maintain_place_pos", category="place",
                         operation="maintain", previous_trial=toa.START, iti_trs=59, baseline_evidence=0.6, baseline_off_category_evidence=0.45,
                         baseline_item_evidence=0.7))
    return pd.DataFrame(rows)


class TestEvidenceByIti:
    def test_baseline_by_trial_also_returns_off_category_and_item_evidence(self):
        d = acquisition("1", "a", ORDER)
        early = d["window_index"] < 3
        d.loc[early, "evidence_face"], d.loc[early, "evidence_place"], d["evidence_item"] = 0.2, 0.8, 0.5
        t = toa.baseline_by_trial(d, toa.build_trial_table(d), (0, 1, 2))
        face = t["category"] == "face"
        assert np.allclose(t.loc[face, ["baseline_evidence", "baseline_off_category_evidence"]], [0.2, 0.8])
        assert np.allclose(t.loc[~face, ["baseline_evidence", "baseline_off_category_evidence"]], [0.8, 0.2])
        assert np.allclose(t["baseline_item_evidence"], 0.5)

    def test_item_evidence_is_nan_when_the_table_has_no_item_column(self):
        d = acquisition("1", "a", ORDER)
        t = toa.baseline_by_trial(d, toa.build_trial_table(d), (0, 1, 2))
        assert t["baseline_item_evidence"].isna().all() and t["baseline_off_category_evidence"].notna().all()

    def test_table_follows_the_planted_effects(self):
        b = toa.baseline_evidence_by_iti(fake_iti_table()).set_index(["predecessor", "iti_trs"])
        diff, same = b.loc["different category"], b.loc["same category"]
        assert diff["self_mean"].is_monotonic_increasing and diff["off_category_mean"].is_monotonic_decreasing
        assert same["self_mean"].nunique() == 1 and same["off_category_mean"].is_monotonic_increasing
        assert diff.loc[5, "self_mean"] == pytest.approx(0.30) and diff.loc[5, "off_category_mean"] == pytest.approx(0.68)
        assert (b["n_subjects"] == 8).all() and (b["n_trials"] == 8 * 15).all()

    def test_first_trials_and_sparse_itis_are_left_out(self):
        t = fake_iti_table()
        b = toa.baseline_evidence_by_iti(t)
        assert 59 not in set(b["iti_trs"])                           # the first-trial lead-in is not an ITI
        assert len(toa.baseline_evidence_by_iti(t, min_trials=10_000)) == 0

    @staticmethod
    def _start_block(subject, boldfile, n_trs, face, place, item, early=0.99):
        """A trial_index-0 start block of n_trs TRs: the last 5 carry (face, place, item), earlier ones `early`."""
        rows = [dict(subject=subject, boldfile=boldfile, trial_index=0, window_index=w,
                     evidence_face=face if w >= n_trs - 5 else early,
                     evidence_place=place if w >= n_trs - 5 else early,
                     evidence_item=item if w >= n_trs - 5 else early) for w in range(n_trs)]
        rows.append(dict(subject=subject, boldfile=boldfile, trial_index=1, window_index=0,
                         evidence_face=0.0, evidence_place=0.0, evidence_item=0.0))      # a trial row: must be ignored
        return pd.DataFrame(rows)

    def test_start_block_uses_only_the_last_five_trs_of_13_tr_blocks(self):
        d = pd.concat([self._start_block("1", "a", 13, 0.2, 0.4, 0.6), self._start_block("2", "b", 13, 0.4, 0.6, 0.8)])
        r = toa.start_block_evidence(d, 13, 5)
        assert r["face"] == pytest.approx(0.3) and r["place"] == pytest.approx(0.5) and r["item"] == pytest.approx(0.7)
        assert r["n_acquisitions"] == 2 and r["n_subjects"] == 2 and r["last_trs"] == 5

    def test_start_block_ignores_blocks_of_other_lengths(self):
        d = pd.concat([self._start_block("1", "a", 13, 0.2, 0.2, 0.2), self._start_block("2", "b", 59, 0.9, 0.9, 0.9)])
        r = toa.start_block_evidence(d, 13, 5)
        assert r["face"] == pytest.approx(0.2) and r["n_acquisitions"] == 1
        assert toa.start_block_evidence(d, 59, 5)["face"] == pytest.approx(0.9)

    def test_start_block_averages_acquisitions_within_subject_first(self):
        d = pd.concat([self._start_block("1", "a", 13, 0.2, 0.2, 0.2), self._start_block("1", "b", 13, 0.4, 0.4, 0.4),
                       self._start_block("2", "c", 13, 0.9, 0.9, 0.9)])
        assert toa.start_block_evidence(d)["face"] == pytest.approx((0.3 + 0.9) / 2)

    def test_start_block_with_no_matching_block_is_nan_not_an_error(self):
        r = toa.start_block_evidence(self._start_block("1", "a", 59, 0.5, 0.5, 0.5), 13, 5)
        assert r["n_acquisitions"] == 0 and np.isnan(r["face"])

    def test_start_block_table_has_one_row_per_block_length(self):
        d = pd.concat([self._start_block("1", "a", 13, 0.2, 0.2, 0.2), self._start_block("2", "b", 59, 0.9, 0.9, 0.9)])
        t = toa.start_block_table(d, 5)
        assert list(t["block_trs"]) == [13, 59] and list(t["n_acquisitions"]) == [1, 1]

    def test_plot_draws_start_block_reference_lines(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = toa.plot_iti(fake_iti_table(), None, dict(face=0.47, place=0.67, item=0.68, block_trs=13, last_trs=5, n_acquisitions=4))
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any(t.startswith("start block: face") for t in texts) and not any("rest" in t for t in texts)
        plt.close(fig)

    def test_plot_has_two_panels_and_survives_missing_item_evidence(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = fake_iti_table().drop(columns="baseline_item_evidence").assign(baseline_item_evidence=np.nan)
        fig = toa.plot_iti(t)
        assert len(fig.axes) == 2
        labels = [txt.get_text() for txt in fig.axes[0].get_legend().get_texts()]
        assert any("current category" in x for x in labels) and any("off category" in x for x in labels) and not any("item" in x for x in labels)
        plt.close(fig)


    def test_plot_with_no_usable_itis_draws_empty_panels_not_an_error(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = toa.plot_iti(fake_iti_table(per_cell=1))               # every ITI has fewer than 10 trials
        assert len(fig.axes) == 2
        plt.close(fig)
