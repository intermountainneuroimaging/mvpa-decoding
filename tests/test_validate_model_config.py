"""validate_model_config.py: structural + data-driven validation of the
model_conditions section."""

import pandas as pd

from utils.validate_model_config import validate_config


def _minimal_config():
    return {
        "model_conditions": {
            "training": {"conditions": {
                "face": {"column": "trial_type", "match": "exact", "value": "face"},
                "place": {"column": "trial_type", "match": "exact", "value": "place"},
            }},
            "testing": {"conditions": {
                "face": {"column": "trial_type", "match": "exact", "value": "face"},
                "place": {"column": "trial_type", "match": "exact", "value": "place"},
            }},
            "timecourse_decoding": {
                "conditions": {
                    "face": {"column": "trial_type", "match": "exact", "value": "face"},
                    "place": {"column": "trial_type", "match": "exact", "value": "place"},
                },
                "window": {
                    "start": {"reference": "onset", "offset_seconds": 0},
                    "end": {"reference": "offset_end", "offset_seconds": 10},
                },
            },
        }
    }


class TestStructuralValidation:
    def test_missing_model_conditions_is_one_error(self):
        errors, warnings = validate_config({})
        assert len(errors) == 1
        assert warnings == []

    def test_valid_minimal_config_no_errors(self):
        errors, warnings = validate_config(_minimal_config())
        assert errors == []
        assert warnings == []

    def test_missing_required_section(self):
        cfg = _minimal_config()
        del cfg["model_conditions"]["testing"]
        errors, _ = validate_config(cfg)
        assert any("testing" in e for e in errors)

    def test_timecourse_decoding_missing_window_is_error(self):
        cfg = _minimal_config()
        del cfg["model_conditions"]["timecourse_decoding"]["window"]
        errors, _ = validate_config(cfg)
        assert any("window" in e for e in errors)

    def test_empty_conditions_object_is_error(self):
        cfg = _minimal_config()
        cfg["model_conditions"]["training"]["conditions"] = {}
        errors, _ = validate_config(cfg)
        assert any("conditions" in e for e in errors)

    def test_invalid_query_inside_condition_is_error(self):
        cfg = _minimal_config()
        cfg["model_conditions"]["training"]["conditions"]["face"] = {"column": "trial_type", "match": "bogus"}
        errors, _ = validate_config(cfg)
        assert len(errors) >= 1

    def test_cross_section_name_mismatch_is_warning(self):
        cfg = _minimal_config()
        cfg["model_conditions"]["testing"]["conditions"] = {
            "face": {"column": "trial_type", "match": "exact", "value": "face"},
        }
        _, warnings = validate_config(cfg)
        assert any("differ" in w for w in warnings)


class TestDataDrivenValidation:
    def _df(self):
        return pd.DataFrame({"trial_type": ["face", "face", "place", "place"]})

    def test_zero_row_match_is_error(self):
        cfg = _minimal_config()
        cfg["model_conditions"]["training"]["conditions"]["place"] = {
            "column": "trial_type", "match": "exact", "value": "nonexistent",
        }
        errors, _ = validate_config(cfg, valid_columns={"trial_type"}, df=self._df())
        assert any("0 rows" in e for e in errors)

    def test_overlapping_conditions_is_warning(self):
        cfg = _minimal_config()
        # both conditions match every row -> full overlap
        cfg["model_conditions"]["training"]["conditions"] = {
            "face": {"column": "trial_type", "match": "regex", "value": ".*"},
            "place": {"column": "trial_type", "match": "regex", "value": ".*"},
        }
        cfg["model_conditions"]["testing"]["conditions"] = cfg["model_conditions"]["training"]["conditions"]
        cfg["model_conditions"]["timecourse_decoding"]["conditions"] = cfg["model_conditions"]["training"]["conditions"]
        _, warnings = validate_config(cfg, valid_columns={"trial_type"}, df=self._df())
        assert any("overlap" in w for w in warnings)

    def test_unknown_column_is_error(self):
        cfg = _minimal_config()
        cfg["model_conditions"]["training"]["conditions"]["face"] = {
            "column": "not_a_real_column", "match": "exact", "value": "face",
        }
        errors, _ = validate_config(cfg, valid_columns={"trial_type"}, df=self._df())
        assert any("unknown column" in e for e in errors)
