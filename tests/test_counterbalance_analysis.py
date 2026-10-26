"""counterbalance_analysis: trial table from event rows, transition tables, back-to-back repeats."""

import pandas as pd

from _interactive_notebooks import counterbalance_analysis as cb


def _events(sequence, subject="1", run=1, task="WMpos"):
    """Events of one acquisition: each trial is a view cue then its operation event."""
    rows, t = [], 0.0
    for i, trial in enumerate(sequence):
        category = trial.split("_")[1]
        rows.append(dict(trial_type=f"view_{category}", onset=t, duration=2.7, event_index=2 * i + 1))
        rows.append(dict(trial_type=trial, onset=t + 2.7, duration=2.7, event_index=2 * i + 2))
        t += 8.0
    df = pd.DataFrame(rows)
    return df.assign(subject=subject, session="A1", task=task, run=run)


def test_trial_table_previous_and_repeats():
    ev = _events(["maintain_face", "maintain_place", "suppress_face"])
    trials = cb.build_trials(ev, ["maintain", "suppress"])
    assert list(trials["previous_trial"]) == [cb.START, "maintain_face", "maintain_place"]
    assert list(trials["same_operation"].fillna(-1)) == [-1, 1, 0]
    assert list(trials["same_category"].fillna(-1)) == [-1, 0, 0]
    assert trials["iti_s"].iloc[1] == 8.0 - 5.4


def test_infer_operations_ignores_image_classes():
    ev = pd.concat([_events(["maintain_face", "track_place"]),
                    pd.DataFrame(dict(trial_type=["positive_face", "negative_place"], onset=[1, 2], duration=1,
                                      event_index=[1, 2], subject="1", session="A1", task="WMpos", run=2))])
    assert cb.infer_operations(ev) == ["maintain", "track"]


def test_transition_counts_and_independence():
    seq = ["a_face", "b_face", "a_face", "b_face", "a_face"]
    trials = cb.build_trials(_events(seq), ["a", "b"])
    counts = cb.transitions(trials, "previous_operation", "operation", [cb.START, "a", "b"], ["a", "b"])
    assert counts.loc["a", "b"] == 2 and counts.loc["b", "a"] == 2 and counts.loc["a", "a"] == 0
    assert counts.loc[cb.START, "a"] == 1
    exp, resid = cb.expected_and_residuals(counts)
    assert exp.loc["a", "b"] == 1.0
    assert cb.independence(counts)["cramers_v"] == 1.0


def test_dropping_cost_counts_back_to_back():
    trials = cb.build_trials(_events(["a_face", "a_face", "b_face"]), ["a", "b"])
    cost = cb.dropping_cost(trials).set_index("operation")
    assert cost.loc["a", "trials"] == 1 and cost.loc["a", "after_dropping_repeats"] == 0
    assert cost.loc["b", "after_dropping_repeats"] == 1
