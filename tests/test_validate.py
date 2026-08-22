"""The allowlist between a language model and a shell command.

Everything the model returns ends up in a command the user pastes into a terminal, so
these tests are as much about what is *refused* as about what is kept.
"""

from __future__ import annotations

import pytest

from app.advice.validate import (
    ALLOWED_PARAMS,
    CRF_LIMITS,
    MAX_CRF_DRIFT,
    MAX_PARAMS,
    PROTECTED_PARAMS,
    Choice,
    FloatRange,
    IntRange,
    Pattern,
    validate_crf,
    validate_params,
    validate_preset,
    validate_tune,
)
from app.models import Encoder

# --- invariants -------------------------------------------------------------


def test_every_encoder_has_an_allowlist_and_a_protected_set() -> None:
    assert set(ALLOWED_PARAMS) == set(Encoder)
    assert set(PROTECTED_PARAMS) == set(Encoder)
    assert set(CRF_LIMITS) == set(Encoder)


@pytest.mark.parametrize("encoder", list(Encoder))
def test_protected_parameters_are_a_subset_of_the_allowlist(encoder: Encoder) -> None:
    # A protected name that is not allowed could never be restored after validation.
    assert PROTECTED_PARAMS[encoder] <= set(ALLOWED_PARAMS[encoder])


@pytest.mark.parametrize("encoder", list(Encoder))
def test_every_allowlisted_name_would_pass_its_own_name_check(encoder: Encoder) -> None:
    # _SAFE_NAME is applied to the model's key before the allowlist is consulted, so a
    # legitimate parameter spelled with a capital or an underscore would be
    # unreachable — the entry would sit in the table and never match.
    for name in ALLOWED_PARAMS[encoder]:
        result = validate_params(encoder, {name: "0"})
        assert not any("not a parameter name" in message for message in result.rejected), name


# --- parameters -------------------------------------------------------------


def test_recognised_parameters_in_range_are_kept() -> None:
    result = validate_params(Encoder.X265, {"aq-strength": "0.9", "rc-lookahead": "60", "sao": "0"})

    assert result.params == {"aq-strength": "0.9", "rc-lookahead": "60", "sao": "0"}
    assert result.rejected == []


def test_names_are_normalised_before_the_lookup() -> None:
    result = validate_params(Encoder.X265, {" AQ-Mode ": " 3 "})

    assert result.params == {"aq-mode": "3"}
    assert result.rejected == []


def test_an_unknown_parameter_is_dropped_and_named() -> None:
    result = validate_params(Encoder.X265, {"enable-warp-motion": "1"})

    assert result.params == {}
    # SVT-AV1's flag on x265: plausible, wrong, and fatal twenty minutes into a run.
    assert result.rejected == ["enable-warp-motion (not a recognised x265 parameter)"]


def test_the_grain_knobs_the_prompt_asks_about_are_all_reachable() -> None:
    # The prompt names these; if the allowlist did not have them, every answer the
    # model gave about grain tuning would be silently thrown away.
    wanted = {
        "film-grain": "10",
        "film-grain-denoise": "0",
        "enable-restoration": "0",
        "enable-cdef": "0",
        "qp-scale-compress-strength": "2",
        "luminance-qp-bias": "20",
        "enable-variance-boost": "1",
        "variance-boost-strength": "2",
        "variance-octile": "6",
        "variance-boost-curve": "1",
        "tf-strength": "1",
        "sharpness": "1",
    }
    result = validate_params(Encoder.SVT_AV1, wanted)

    assert result.params == wanted
    assert result.rejected == []


def test_enable_hdr_stays_out_because_svt_av1_does_not_have_it() -> None:
    # Measured: SVT-AV1 4.2.0 answers "Error parsing option enable-hdr", exactly as it
    # does for an invented key. HDR is signalled through the colour parameters instead.
    result = validate_params(Encoder.SVT_AV1, {"enable-hdr": "1"})

    assert result.params == {}
    assert result.rejected == ["enable-hdr (not a recognised svt-av1 parameter)"]


def test_a_parameter_from_the_other_encoder_is_dropped() -> None:
    assert validate_params(Encoder.SVT_AV1, {"aq-strength": "0.9"}).params == {}
    assert validate_params(Encoder.X265, {"film-grain": "12"}).params == {}


@pytest.mark.parametrize(
    ("encoder", "name", "value"),
    [
        (Encoder.SVT_AV1, "film-grain", "60"),  # 0-50
        # Ranges measured against SVT-AV1 4.2.0's own refusals.
        (Encoder.SVT_AV1, "tune", "9"),  # 0-2
        (Encoder.SVT_AV1, "tune", "3"),  # 3 aborts a random-access encode outright
        (Encoder.SVT_AV1, "qp-scale-compress-strength", "4"),  # 0-3
        (Encoder.SVT_AV1, "luminance-qp-bias", "101"),  # 0-100
        (Encoder.SVT_AV1, "variance-boost-strength", "0"),  # 1-4
        (Encoder.SVT_AV1, "variance-boost-strength", "5"),  # 1-4
        (Encoder.SVT_AV1, "variance-octile", "9"),  # 1-8
        (Encoder.SVT_AV1, "variance-boost-curve", "3"),  # 0-2
        (Encoder.SVT_AV1, "tf-strength", "5"),  # 0-4
        (Encoder.X265, "aq-strength", "9.5"),  # 0.0-3.0
        (Encoder.X265, "qcomp", "0.1"),  # 0.5-1.0
        (Encoder.X265, "rd", "0"),  # 1-6
        (Encoder.X265, "sao", "2"),  # boolean
    ],
)
def test_out_of_range_values_are_dropped(encoder: Encoder, name: str, value: str) -> None:
    result = validate_params(encoder, {name: value})

    assert result.params == {}
    assert result.rejected == [f"{name}={value} (out of range)"]


def test_a_non_numeric_value_for_a_numeric_parameter_is_dropped() -> None:
    result = validate_params(Encoder.X265, {"keyint": "auto"})

    assert result.params == {}
    assert "out of range" in result.rejected[0]


@pytest.mark.parametrize(
    "value",
    [
        "0; rm -rf /",
        "$(whoami)",
        "`id`",
        "0 && curl http://example.com",
        "0\nsao=1",
        "'",
        "",
        "   ",
        "0" * 200,
    ],
)
def test_shell_metacharacters_never_reach_a_parameter_value(value: str) -> None:
    result = validate_params(Encoder.X265, {"sao": value})

    assert result.params == {}
    assert result.rejected == ["sao (unsafe or empty value)"]


@pytest.mark.parametrize("name", ["SAO;", "--sao", "sao sao", "9lives", "", "a" * 60])
def test_a_value_that_is_not_a_parameter_name_is_dropped(name: str) -> None:
    result = validate_params(Encoder.X265, {name: "0"})

    assert result.params == {}
    assert result.rejected == [f"{name!r} (not a parameter name)"]


def test_the_number_of_parameters_is_capped_and_the_excess_reported() -> None:
    # A model looping on itself should not produce a 5,000-character params string.
    params = {f"param-{index}": "1" for index in range(MAX_PARAMS + 5)}

    result = validate_params(Encoder.X265, params)

    assert result.params == {}
    assert result.rejected[-1] == "5 further parameters (over the limit)"
    # Only MAX_PARAMS names were examined, plus the summary line.
    assert len(result.rejected) == MAX_PARAMS + 1


def test_hdr_metadata_shapes_are_matched_exactly() -> None:
    good = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    assert validate_params(Encoder.X265, {"master-display": good}).params == {
        "master-display": good
    }

    # x265 wants integers in 0.00002 units; SVT-AV1's float form is not valid here.
    floats = "G(0.265,0.69)B(0.15,0.06)R(0.68,0.32)WP(0.3127,0.329)L(1000,0.0001)"
    assert validate_params(Encoder.X265, {"master-display": floats}).params == {}
    # ...and the float form is what SVT-AV1 takes.
    assert validate_params(Encoder.SVT_AV1, {"mastering-display": floats}).params == {
        "mastering-display": floats
    }
    assert validate_params(Encoder.SVT_AV1, {"mastering-display": "G(1,2)"}).params == {}


def test_light_level_pairs_need_both_numbers() -> None:
    assert validate_params(Encoder.X265, {"max-cll": "1000,400"}).params == {"max-cll": "1000,400"}
    assert validate_params(Encoder.X265, {"max-cll": "1000"}).params == {}


def test_colour_names_are_checked_against_the_code_tables() -> None:
    assert validate_params(Encoder.X265, {"colorprim": "bt2020"}).params == {"colorprim": "bt2020"}
    # A plausible spelling that x265 would reject outright.
    assert validate_params(Encoder.X265, {"colorprim": "bt.2020"}).params == {}
    assert validate_params(Encoder.X265, {"range": "full"}).params == {"range": "full"}
    assert validate_params(Encoder.X265, {"range": "tv"}).params == {}


def test_deblock_takes_one_value_because_the_params_string_is_colon_joined() -> None:
    # x265 applies a single value to both the tC and beta offsets.
    assert validate_params(Encoder.X265, {"deblock": "-1"}).params == {"deblock": "-1"}
    # "-1:-1" is how x265's own documentation writes it, and it is unusable here:
    # params_string joins on ':', so libx265 reads "-1" as a parameter name
    # ("Unknown option: -1:sao") and drops whatever parameter followed it.
    assert validate_params(Encoder.X265, {"deblock": "-1:-1"}).params == {}
    assert validate_params(Encoder.X265, {"deblock": "loose"}).params == {}


def test_no_value_may_contain_a_colon_whatever_the_parameter() -> None:
    result = validate_params(Encoder.SVT_AV1, {"keyint": "240:1"})

    assert result.params == {}
    assert result.rejected == ["keyint (unsafe or empty value)"]
    # And nothing legitimate needs one — not even the punctuation-heavy HDR values.
    hdr = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)"
    assert validate_params(Encoder.SVT_AV1, {"mastering-display": hdr}).params
    assert validate_params(Encoder.X265, {"max-cll": "1000,400"}).params


def test_level_patterns() -> None:
    assert validate_params(Encoder.X265, {"level-idc": "5.1"}).params == {"level-idc": "5.1"}
    assert validate_params(Encoder.X265, {"level-idc": "150"}).params == {"level-idc": "150"}


# --- rule types -------------------------------------------------------------


def test_rule_types_reject_what_they_cannot_parse() -> None:
    assert IntRange(0, 10).accepts("5")
    assert not IntRange(0, 10).accepts("5.5")
    assert not IntRange(0, 10).accepts("x")

    assert FloatRange(0.0, 1.0).accepts("0.5")
    assert not FloatRange(0.0, 1.0).accepts("2")

    assert Choice(frozenset({"grain"})).accepts("GRAIN")
    assert not Choice(frozenset({"grain"})).accepts("grainy")

    import re

    assert Pattern(re.compile(r"\d+")).accepts("42")
    # fullmatch, not search: a valid prefix is not a valid value.
    assert not Pattern(re.compile(r"\d+")).accepts("42px")


# --- CRF --------------------------------------------------------------------


def test_a_crf_close_to_the_baseline_is_accepted_silently() -> None:
    assert validate_crf(Encoder.X265, 22.0, baseline=20.0) == (22.0, None)
    assert validate_crf(Encoder.SVT_AV1, 30.5, baseline=28.0) == (30.5, None)


def test_a_crf_far_from_the_baseline_is_pulled_back_with_an_explanation() -> None:
    value, note = validate_crf(Encoder.X265, 30.0, baseline=20.0)

    assert value == 24.0  # baseline + MAX_CRF_DRIFT
    assert note is not None
    assert "pulled back to 24" in note
    assert "4 points" in note


def test_the_drift_bound_applies_in_both_directions() -> None:
    value, note = validate_crf(Encoder.X265, 12.0, baseline=20.0)

    assert value == 16.0
    assert note is not None


def test_a_crf_outside_the_encoders_range_is_clamped_first() -> None:
    # SVT-AV1 tops out at 55 here; a hallucinated 63 must not become baseline+4.
    value, note = validate_crf(Encoder.SVT_AV1, 63.0, baseline=52.0)

    assert value == 55.0
    assert note is not None


def test_exactly_at_the_drift_limit_is_allowed() -> None:
    assert validate_crf(Encoder.X265, 20.0 + MAX_CRF_DRIFT, baseline=20.0) == (24.0, None)


def test_crf_is_rounded_to_one_decimal() -> None:
    value, _ = validate_crf(Encoder.X265, 20.1234, baseline=20.0)
    assert value == 20.1


# --- presets and tunes ------------------------------------------------------


@pytest.mark.parametrize("preset", ["0", "4", "13"])
def test_numeric_svt_presets_are_accepted(preset: str) -> None:
    assert validate_preset(Encoder.SVT_AV1, preset, baseline="4") == (preset, None)


@pytest.mark.parametrize("preset", ["14", "-1", "slow", "", "4.5"])
def test_a_non_svt_preset_falls_back_to_the_baseline(preset: str) -> None:
    value, note = validate_preset(Encoder.SVT_AV1, preset, baseline="4")

    assert value == "4"
    assert note is not None and "0–13" in note


@pytest.mark.parametrize("preset", ["slow", "SLOWER", "placebo"])
def test_named_x265_presets_are_accepted_case_insensitively(preset: str) -> None:
    value, note = validate_preset(Encoder.X265, preset, baseline="slow")

    assert value == preset.casefold()
    assert note is None


@pytest.mark.parametrize("preset", ["5", "sloww", "veryveryslow"])
def test_a_non_x265_preset_falls_back_to_the_baseline(preset: str) -> None:
    value, note = validate_preset(Encoder.X265, preset, baseline="slow")

    assert value == "slow"
    assert note is not None


def test_x265_tunes_are_checked_against_the_real_list() -> None:
    assert validate_tune(Encoder.X265, "grain") == ("grain", None)
    assert validate_tune(Encoder.X265, "GRAIN") == ("grain", None)
    assert validate_tune(Encoder.X265, None) == (None, None)
    assert validate_tune(Encoder.X265, "  ") == (None, None)

    value, note = validate_tune(Encoder.X265, "film")
    assert value is None
    assert note is not None and "not an x265 tune" in note


def test_svt_av1_takes_no_named_tune() -> None:
    # Its tune is numeric and belongs in the parameter string; passing -tune 0 to
    # libsvtav1 through ffmpeg is not the same thing.
    value, note = validate_tune(Encoder.SVT_AV1, "0")

    assert value is None
    assert note is not None and "as a parameter" in note
