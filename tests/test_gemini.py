"""Gemini deciding the settings, and never as a single point of failure.

Three separable things are tested here: what the model is *told* (``build_context``),
what is made of what it *says* (``advice_from``), and what happens when the call itself
goes wrong (``GeminiClient.decide``, which must never raise).

The inversion these tests are written around: nothing proposes settings before the model
is asked. There is no baseline in the context and none behind the answer, so an omitted
field is a *missing* field, and an answer with nothing usable in it is a stated problem
rather than a table quietly showing through. The tables are still there — as the labelled
fallback :mod:`app.advice.pipeline` reaches for — but they are not on this path.

The technical-specs look-up is the one call whose failure the user does see, because
without rows there is nothing to show in the panel — so it raises rather than pretends,
and everything it does return is labelled with where it came from.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

from app.advice.rules import REAL_GRAIN_CRF_CEILING
from app.models import (
    AdviceSource,
    Encoder,
    GrainLevel,
    SourceTool,
    SpecsSource,
    TechnicalSpecs,
)
from app.providers import ProviderDisabled, ProviderUnavailable
from app.providers.gemini import (
    ACCOUNT_TOLERANCE,
    MAX_ADJUSTMENTS,
    MAX_NOTES,
    MAX_OVERVIEW_CHARS,
    NO_USABLE_PLAN,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    TECHNICAL_PROMPT,
    TECHNICAL_SCHEMA,
    GeminiClient,
    advice_from,
    build_context,
)
from app.sources import parse_source
from tests.support import (
    Handler,
    fixture,
    gemini_answer,
    make_settings,
    media,
    mock_client,
    movie,
    recording_handler,
    request_for,
    run,
    video_track,
)

API_KEY = "AIzaSyTestKeyNotReal"


def plan(encoder: str, crf: float, preset: str = "4", **extra: Any) -> dict[str, Any]:
    """One plan row, with an account of its CRF that adds up.

    A plan whose ``adjustments`` do not total its CRF is a reported fault in its own
    right, so tests about anything else say ``plan(...)`` and keep that fault out of the
    way. Tests about the account itself pass ``adjustments`` explicitly.
    """
    row: dict[str, Any] = {
        "encoder": encoder,
        "crf": crf,
        "preset": preset,
        "adjustments": [{"label": "the number for this film", "delta": crf}],
        "params": [],
        "rationale": [],
    }
    return {**row, **extra}


def envelope(answer: Any, **candidate_overrides: Any) -> dict[str, Any]:
    """Gemini's ``generateContent`` response around one JSON answer."""
    text = answer if isinstance(answer, str) else json.dumps(answer)
    candidate: dict[str, Any] = {
        "content": {"role": "model", "parts": [{"text": text}]},
        "finishReason": "STOP",
    }
    return {"candidates": [{**candidate, **candidate_overrides}]}


def gemini(handler: Handler, **overrides: Any) -> GeminiClient:
    settings = make_settings(**{"gemini_api_key": API_KEY, **overrides})
    return GeminiClient(settings, client=mock_client(handler))


def answering(answer: Any) -> tuple[GeminiClient, list[httpx2.Request]]:
    handler, seen = recording_handler(httpx2.Response(200, json=envelope(answer)))
    return gemini(handler), seen


def decided(payload: dict[str, Any], **overrides: Any) -> Any:
    """The advice built from a payload, asserting there was one to build."""
    decision = advice_from(payload, request_for(**overrides))
    assert decision.problem is None, decision.problem
    assert decision.advice is not None
    return decision.advice


# --- what the model is told --------------------------------------------------


def test_the_local_file_path_is_never_sent() -> None:
    request = request_for(input_path="/home/nick/rips/Blade Runner (1982).mkv")

    body = json.dumps(build_context(request))

    # The model has no use for a path on someone's NAS, and no business seeing it.
    assert "/home/nick" not in body
    assert "input_path" not in body
    assert "output_stem" not in body
    assert "blade-runner-1982-1080p" not in body


def test_the_api_key_is_never_in_the_context() -> None:
    assert API_KEY not in json.dumps(build_context(request_for()))


def test_no_settings_of_any_kind_reach_the_model() -> None:
    """The heart of it: an answer to react to is an anchor, so none is sent.

    Checked by key rather than by value, because a CRF that leaked in as ``28.0`` would
    read as a plausible number in a body full of numbers — while a key named ``crf``
    anywhere in the context means something upstream decided one.
    """
    context = build_context(request_for())

    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {key for value in node.values() for key in keys(value)}
        if isinstance(node, list):
            return {key for value in node for key in keys(value)}
        return set()

    assert {"proposal", "plans", "crf", "preset", "params", "adjustments"} & keys(context) == set()
    assert "baseline" not in json.dumps(context).casefold()


def test_the_context_carries_the_film_and_the_facts() -> None:
    context = build_context(request_for())

    assert context["film"]["title"] == "Blade Runner"
    assert context["film"]["year"] == 1982
    assert context["film"]["directors"] == ["Ridley Scott"]
    assert context["imdb_technical"]["negative_formats"] == ["35 mm"]
    assert context["preferences"] == {"speed": "balanced", "size": "balanced"}
    # Cited by the prompt, so it travels with the facts rather than being implied.
    assert context["limits"]["svt_av1_real_grain_crf_ceiling"] == REAL_GRAIN_CRF_CEILING


def test_the_grain_estimate_goes_as_a_guess_and_says_so() -> None:
    context = build_context(request_for())

    estimate = context["grain_estimate"]
    assert estimate["level"] == "moderate"
    assert estimate["origin_format"] == "Super 35 mm"
    assert estimate["photochemical"] is True
    assert estimate["set_by_user"] is False
    # Labelled: the model is asked to weigh this, not to ratify it.
    assert "a guess to weigh, not a decision" in estimate["made_by"]


def test_a_hand_set_grain_level_is_marked_as_the_users() -> None:
    """The one thing in the context the model is told it may not overturn."""
    estimate = build_context(request_for(grain_override=GrainLevel.HEAVY))["grain_estimate"]

    assert estimate["level"] == "heavy"
    assert estimate["set_by_user"] is True
    assert estimate["confidence"] == 1.0


def test_the_context_describes_the_source_file() -> None:
    request = request_for(
        source=parse_source(fixture("ffprobe_uhd_hdr.json")).media, source_tool=SourceTool.FFPROBE
    )

    source = build_context(request)["source"]

    assert source["resolution_class"] == "2160p"
    assert source["parsed_from"] == "ffprobe"
    assert source["video"]["codec"] == "hevc"
    assert source["video"]["bit_depth"] == 10
    assert source["video"]["hdr"] is True
    assert source["video"]["has_mastering_display"] is True
    assert source["video"]["max_cll"] == 1000
    # Four decimal places is plenty for a judgement about compression history.
    assert source["video"]["bits_per_pixel"] == pytest.approx(0.2923, abs=5e-5)
    # keyint is the model's to set, so it is given the arithmetic rather than the rule:
    # ten seconds at 23.976 fps, rounded.
    assert source["video"]["ten_second_keyframe_interval"] == 240
    assert [track["codec"] for track in source["audio"]] == ["truehd", "ac3"]
    assert source["subtitle_count"] == 1


def test_absent_facts_are_omitted_rather_than_sent_as_null() -> None:
    # No film, no technical rows, no video track: a null-strewn context invites the
    # model to fill the gaps in, which is exactly what it must not do here.
    request = request_for(movie=None, specs=TechnicalSpecs(), source=media(video=None, audio=[]))

    context = build_context(request)

    assert "film" not in context
    assert "imdb_technical" not in context
    assert "video" not in context["source"]
    assert "audio" not in context["source"]
    assert "release_year" not in context


def test_a_parsed_year_reaches_the_model_when_no_film_was_looked_up() -> None:
    # Without a TMDB key the CLI has only the year in the filename, and that year is
    # the whole input to the older-film half of the grain reasoning.
    context = build_context(request_for(movie=None, fallback_year=1962))

    assert "film" not in context
    assert context["release_year"] == 1962


def test_a_long_overview_is_truncated() -> None:
    request = request_for(movie=movie(overview="x" * 2_000))

    assert len(build_context(request)["film"]["overview"]) == MAX_OVERVIEW_CHARS


def test_at_most_eight_audio_tracks_are_described() -> None:
    # A dozen dubs is a real disc; describing all of them wastes the context window.
    request = request_for(source=media(audio=media().audio * 12))

    assert len(build_context(request)["source"]["audio"]) == 8


# --- the contract -----------------------------------------------------------


def test_the_response_schema_avoids_what_gemini_rejects() -> None:
    # Gemini's responseSchema is an OpenAPI 3.0 subset: additionalProperties is not in
    # it, which is why the parameter map is carried as name/value pairs.
    def walk(node: object) -> None:
        if isinstance(node, dict):
            assert "additionalProperties" not in node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(RESPONSE_SCHEMA)
    assert RESPONSE_SCHEMA["required"] == ["summary", "grain", "plans"]


def test_the_schema_demands_everything_that_has_no_fallback() -> None:
    """Whatever the model omits is simply missing, so the schema asks for all of it."""
    required = RESPONSE_SCHEMA["properties"]["plans"]["items"]["required"]

    assert required == ["encoder", "crf", "preset", "params", "adjustments", "rationale"]
    assert RESPONSE_SCHEMA["properties"]["grain"]["required"] == [
        "level",
        "confidence",
        "reasons",
    ]


def test_the_system_prompt_never_offers_a_proposal_to_edit() -> None:
    """The prompt used to say parameters left out kept their baseline values. Nothing does."""
    lowered = SYSTEM_PROMPT.casefold()

    assert "baseline" not in lowered
    assert "rules engine" not in lowered
    # The only mention of a proposal is the sentence saying there is not one.
    assert lowered.count("proposal") == 1
    assert "There is no proposal to edit" in SYSTEM_PROMPT
    assert "what you return is what the user encodes with" in SYSTEM_PROMPT


def test_the_system_prompt_asks_for_a_complete_parameter_set() -> None:
    assert "Return the complete parameter set for each encoder" in SYSTEM_PROMPT
    assert "a parameter you omit is absent from the command line" in SYSTEM_PROMPT
    # The two that are easy to forget precisely because something used to supply them.
    assert "keyint, and SVT-AV1's tune" in SYSTEM_PROMPT


def test_the_system_prompt_asks_the_model_to_account_for_its_crf() -> None:
    assert "the named factors that add up to `crf`" in SYSTEM_PROMPT
    assert "the total must equal `crf`" in SYSTEM_PROMPT
    # Why it is asked for at all, which is what keeps the labels meaningful.
    assert "a CRF with no account is a number they cannot argue with" in SYSTEM_PROMPT


def test_the_system_prompt_hands_the_grain_decision_over() -> None:
    assert "Return `grain`" in SYSTEM_PROMPT
    assert "you may disagree with it" in SYSTEM_PROMPT
    assert "set_by_user, the level is the user's and is not yours to change" in SYSTEM_PROMPT


def test_the_system_prompt_states_the_limits_it_will_be_held_to() -> None:
    # Anything the validator silently enforces should be in the prompt too, or the
    # model spends its output on suggestions that get thrown away.
    assert "10-55 for SVT-AV1" in SYSTEM_PROMPT
    assert "slower" in SYSTEM_PROMPT  # the x265 preset ladder
    assert "film-grain" in SYSTEM_PROMPT  # the SVT-AV1 parameter allowlist
    assert "will be overwritten with the file's own values" in SYSTEM_PROMPT
    # Every value ends up in one colon-joined string, so the model has to know.
    assert "never deblock=-1:-1" in SYSTEM_PROMPT


def test_the_system_prompt_asks_for_grain_tuning_and_not_only_a_crf() -> None:
    # The complaint this answers: the model returned CRF opinions and no grain
    # parameters at all, so nothing it said reached -svtav1-params.
    assert "film-grain-denoise" in SYSTEM_PROMPT
    assert "not an answer for a film source" in SYSTEM_PROMPT
    for knob in (
        "enable-restoration",
        "qp-scale-compress-strength",
        "luminance-qp-bias",
        "variance-boost-strength",
        "strong-intra-smoothing",
        "psy-rdoq",
    ):
        assert knob in SYSTEM_PROMPT, knob
    # And what tune grain leaves alone, because assuming otherwise is the usual error.
    assert "does not touch qcomp" in SYSTEM_PROMPT


def test_the_system_prompt_asks_for_the_older_film_case_specifically() -> None:
    assert "pre-1970" in SYSTEM_PROMPT
    assert "dupe" in SYSTEM_PROMPT
    assert "the year alone is not it" in SYSTEM_PROMPT


# --- building the advice ----------------------------------------------------


def test_a_complete_answer_becomes_the_advice() -> None:
    advice = decided(gemini_answer())

    assert advice.source is AdviceSource.GEMINI
    assert advice.summary is not None and advice.summary.startswith("A 1982 anamorphic")
    assert advice.notes == ["The opening flyover is the hardest shot in the film for any encoder."]
    assert advice.rejected_flags == []

    svt, x265 = advice.plans
    assert (svt.crf, svt.preset, svt.tune) == (26.0, "3", None)
    assert svt.params["film-grain-denoise"] == "0"
    assert svt.rationale == ["A slower preset pays for itself on the smoke and rain."]
    assert (x265.crf, x265.preset, x265.tune) == (19.0, "slower", "grain")
    assert x265.params["psy-rd"] == "1.5"


def test_the_plans_come_back_in_a_stable_order() -> None:
    """Whichever way round the model answers, the page puts them in the same places."""
    payload = gemini_answer(plans=[plan("x265", 20.0, "slow"), plan("svt-av1", 27.0)])

    advice = decided(payload)

    assert [row.encoder for row in advice.plans] == [Encoder.SVT_AV1, Encoder.X265]


def test_a_parameter_the_model_left_out_is_absent() -> None:
    """The inversion, at the level it matters: no table is showing through the gaps."""
    payload = gemini_answer(
        plans=[plan("svt-av1", 27.0, params=[{"name": "sharpness", "value": "1"}])]
    )

    params = decided(payload).plans[0].params

    assert params["sharpness"] == "1"
    assert "film-grain" not in params
    assert "enable-restoration" not in params
    assert "qp-scale-compress-strength" not in params


def test_the_keyframe_interval_survives_an_answer_that_forgets_it() -> None:
    """Ten seconds of frames is arithmetic, and a command line without it is broken."""
    advice = decided(gemini_answer(plans=[plan("svt-av1", 27.0)]))

    assert advice.plans[0].params["keyint"] == "240"


def test_the_adjustments_are_the_models_own_and_add_up() -> None:
    advice = decided(gemini_answer())
    steps = advice.plans[0].adjustments

    assert [step.label for step in steps] == ["1080p starting point", "heavy grain to code"]
    assert steps[1].detail == "Funds the grain."
    assert sum(step.delta for step in steps) == pytest.approx(advice.plans[0].crf)
    # Nothing synthesised on the model's behalf, and nothing labelled "Gemini" either:
    # the whole account is its own reasoning.
    assert not any(step.label == "Gemini" for step in steps)


def test_an_account_that_does_not_add_up_is_reconciled_and_reported() -> None:
    """The user is shown these as a total, so a table that does not total is a broken one."""
    payload = gemini_answer(
        plans=[plan("svt-av1", 26.0, adjustments=[{"label": "1080p", "delta": 28.0}])]
    )

    advice = decided(payload)
    steps = advice.plans[0].adjustments

    assert steps[-1].label == "unaccounted for"
    assert steps[-1].delta == pytest.approx(-2.0)
    assert sum(step.delta for step in steps) == pytest.approx(26.0)
    assert any("did not add up" in flag or "not the 26" in flag for flag in advice.rejected_flags)


def test_rounding_slack_is_not_reported_as_a_broken_account() -> None:
    payload = gemini_answer(
        plans=[
            plan(
                "svt-av1",
                26.0,
                adjustments=[{"label": "1080p", "delta": 28.0}, {"label": "grain", "delta": -1.98}],
            )
        ]
    )

    advice = decided(payload)

    assert len(advice.plans[0].adjustments) == 2
    assert advice.rejected_flags == ["no x265 plan in the answer"]
    assert ACCOUNT_TOLERANCE > 0.02


def test_no_account_at_all_becomes_one_row_and_a_flag() -> None:
    payload = gemini_answer(plans=[plan("svt-av1", 27.0, adjustments=[])])

    advice = decided(payload)
    (step,) = advice.plans[0].adjustments

    assert (step.label, step.delta) == ("Gemini's CRF for this film", 27.0)
    assert any("no usable factors" in flag for flag in advice.rejected_flags)


def test_a_crf_the_encoder_would_refuse_is_clamped_and_the_row_says_why() -> None:
    payload = gemini_answer(
        plans=[plan("x265", 99.0, "slow", adjustments=[{"label": "starting point", "delta": 99.0}])]
    )

    advice = decided(payload)
    plan_ = advice.plans[0]

    assert plan_.crf == 40.0
    assert any("clamped to 40" in flag for flag in advice.rejected_flags)
    # The account still totals, and the difference is attributed to the real cause
    # rather than to the model's arithmetic.
    assert plan_.adjustments[-1].label == "held to x265's range"
    assert sum(step.delta for step in plan_.adjustments) == pytest.approx(40.0)
    assert not any("did not add up" in flag for flag in advice.rejected_flags)


def test_more_adjustments_than_the_page_will_show_are_capped() -> None:
    payload = gemini_answer(
        plans=[
            plan(
                "svt-av1",
                27.0,
                adjustments=[{"label": f"factor {index}", "delta": 1.0} for index in range(20)],
            )
        ]
    )

    advice = decided(payload)

    assert len(advice.plans[0].adjustments) == MAX_ADJUSTMENTS + 1


def test_the_bitrate_is_estimated_from_the_models_own_numbers() -> None:
    lower = decided(gemini_answer(plans=[plan("svt-av1", 22.0)]))
    higher = decided(gemini_answer(plans=[plan("svt-av1", 34.0)]))

    left = lower.plans[0].estimated_bitrate_bps
    right = higher.plans[0].estimated_bitrate_bps
    assert left is not None and right is not None
    assert left > right


# --- building the advice: the grain -----------------------------------------


def test_the_grain_profile_is_the_models() -> None:
    advice = decided(gemini_answer())

    assert advice.grain.level is GrainLevel.HEAVY  # the estimate for this film says moderate
    assert advice.grain.confidence == 0.85
    assert advice.grain.origin_format == "35 mm"
    assert advice.grain.reasons == [
        "A 1982 anamorphic 35 mm negative, printed rather than scanned clean."
    ]
    assert advice.grain.user_override is False


def test_a_hand_set_grain_level_beats_the_models() -> None:
    """The user asked for a level, so it is not the model's to reconsider."""
    advice = decided(gemini_answer(), grain_override=GrainLevel.LIGHT)

    assert advice.grain.level is GrainLevel.LIGHT
    assert advice.grain.user_override is True
    assert advice.grain.confidence == 1.0


def test_an_unusable_grain_level_falls_back_to_the_estimate_and_says_so() -> None:
    payload = gemini_answer(grain={"level": "quite grainy really", "confidence": 0.9})

    advice = decided(payload)

    assert advice.grain.level is GrainLevel.MODERATE
    assert any("quite grainy really" in flag for flag in advice.rejected_flags)


def test_the_origin_format_falls_back_because_the_grain_rules_hinge_on_it() -> None:
    """Losing it would reclassify a 35 mm negative as maybe-digital, which is not a detail."""
    payload = gemini_answer(grain={"level": "heavy", "confidence": 0.8, "reasons": []})

    advice = decided(payload)

    assert advice.grain.origin_format == "Super 35 mm"
    assert advice.grain.is_photochemical is True


def test_a_confidence_outside_the_scale_is_brought_back_onto_it() -> None:
    payload = gemini_answer(grain={"level": "heavy", "confidence": 4.2, "reasons": ["Sure."]})

    assert decided(payload).grain.confidence == 1.0


# --- building the advice: coherence and the file ----------------------------


def test_a_plan_that_starves_its_own_grain_is_reported_and_left_standing() -> None:
    """The coherence check reports; it does not rewrite. The model may have a reason."""
    payload = gemini_answer(
        plans=[
            plan(
                "svt-av1",
                31.0,
                params=[
                    {"name": "film-grain", "value": "8"},
                    {"name": "film-grain-denoise", "value": "0"},
                ],
            )
        ]
    )

    advice = decided(payload)

    # Untouched: the CRF the model asked for, over the grain handling it asked for.
    assert advice.plans[0].crf == 31.0
    assert advice.plans[0].params["film-grain-denoise"] == "0"
    complaint = advice.warnings[-1]
    assert "film-grain-denoise=0" in complaint
    assert f"above the {REAL_GRAIN_CRF_CEILING:g}" in complaint
    assert "why this film is an exception" in complaint


def test_a_high_crf_over_replaced_grain_is_coherent_and_says_nothing() -> None:
    payload = gemini_answer(
        plans=[
            plan(
                "svt-av1",
                33.0,
                params=[
                    {"name": "film-grain-denoise", "value": "1"},
                    {"name": "film-grain", "value": "10"},
                ],
            )
        ]
    )

    advice = decided(payload)

    assert advice.plans[0].crf == 33.0
    assert advice.warnings == []


def test_coded_grain_under_the_ceiling_says_nothing_either() -> None:
    advice = decided(gemini_answer())

    assert advice.plans[0].params["film-grain-denoise"] == "0"
    assert advice.warnings == []


def test_the_sources_own_colour_metadata_always_wins() -> None:
    hdr = parse_source(fixture("ffprobe_uhd_hdr.json")).media
    payload = gemini_answer(
        plans=[
            plan(
                "x265",
                20.0,
                "slow",
                params=[
                    {"name": "master-display", "value": "G(1,1)B(1,1)R(1,1)WP(1,1)L(1,1)"},
                    {"name": "colorprim", "value": "bt709"},
                    {"name": "max-cll", "value": "4000,600"},
                ],
            )
        ]
    )

    advice = decided(payload, source=hdr)

    # The model has not seen the file; a plausible-looking guess here would silently
    # mis-tag the HDR of the finished encode.
    params = advice.plans[0].params
    assert hdr.video is not None and hdr.video.mastering_display is not None
    assert params["master-display"] == hdr.video.mastering_display.to_x265()
    assert params["colorprim"] == "bt2020"
    assert params["max-cll"] == "1000,400"


def test_a_protected_parameter_the_source_does_not_justify_is_removed() -> None:
    # An SDR source has no mastering display, so an invented one is not "additional
    # metadata", it is a lie about the master.
    payload = gemini_answer(
        plans=[
            plan(
                "x265",
                20.0,
                "slow",
                params=[{"name": "master-display", "value": "G(1,1)B(1,1)R(1,1)WP(1,1)L(1,1)"}],
            )
        ]
    )

    assert "master-display" not in decided(payload).plans[0].params


# --- building the advice: answers that are wrong ----------------------------


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("svt-av1", Encoder.SVT_AV1),
        ("SVT_AV1", Encoder.SVT_AV1),
        ("av1", Encoder.SVT_AV1),
        ("libsvtav1", Encoder.SVT_AV1),
        ("x265", Encoder.X265),
        ("libx265", Encoder.X265),
        ("HEVC", Encoder.X265),
        ("h265", Encoder.X265),
    ],
)
def test_the_encoder_name_is_read_generously(alias: str, expected: Encoder) -> None:
    preset = "4" if expected is Encoder.SVT_AV1 else "slower"
    payload = gemini_answer(plans=[plan(alias, 25.0, preset)])

    advice = decided(payload)
    row = advice.plan_for(expected)

    assert row is not None and (row.crf, row.preset) == (25.0, preset)
    # The alias resolved rather than being read as an encoder we do not run.
    assert not any("unknown encoder" in flag for flag in advice.rejected_flags)


def test_a_plan_for_an_encoder_that_does_not_exist_is_reported() -> None:
    payload = gemini_answer(plans=[plan("svt-av1", 27.0), plan("vp9", 30.0)])

    advice = decided(payload)

    assert "a plan for unknown encoder 'vp9'" in advice.rejected_flags


def test_a_second_plan_for_the_same_encoder_is_ignored() -> None:
    payload = gemini_answer(plans=[plan("x265", 21.0, "slow"), plan("x265", 16.0, "veryslow")])

    advice = decided(payload)

    assert advice.plans[0].crf == 21.0
    assert "a second x265 plan" in advice.rejected_flags


def test_one_encoder_missing_is_kept_and_reported() -> None:
    """Half an answer is still an answer; a table filling the other half would not be."""
    advice = decided(gemini_answer(plans=[plan("svt-av1", 27.0)]))

    assert [row.encoder for row in advice.plans] == [Encoder.SVT_AV1]
    assert advice.rejected_flags == ["no x265 plan in the answer"]


def test_a_crf_that_is_not_a_number_drops_the_whole_plan() -> None:
    """There is nothing to fall back to, and a CRF nobody chose is worse than no plan."""
    payload = gemini_answer(
        plans=[plan("svt-av1", 27.0), {"encoder": "x265", "crf": "twenty", "preset": "slow"}]
    )

    advice = decided(payload)

    assert [row.encoder for row in advice.plans] == [Encoder.SVT_AV1]
    assert any("'twenty' and not a number" in flag for flag in advice.rejected_flags)


@pytest.mark.parametrize("payload", [{}, {"summary": "Reviewed for this film."}, {"plans": []}])
def test_an_answer_with_no_usable_plan_decides_nothing(payload: dict[str, Any]) -> None:
    """The safety net, and the whole of it: no silent blend, one stated problem."""
    decision = advice_from(payload, request_for())

    assert decision.advice is None
    assert decision.problem == NO_USABLE_PLAN


def test_a_preset_the_encoder_does_not_have_is_reported() -> None:
    payload = gemini_answer(plans=[plan("svt-av1", 27.0, "insanely-slow")])

    advice = decided(payload)

    assert any("not an SVT-AV1 preset" in flag for flag in advice.rejected_flags)
    # Not a table's preset: a constant, because the model was never shown one.
    assert advice.plans[0].preset == "5"


def test_an_x265_tune_can_be_cleared_but_svt_av1_gets_none() -> None:
    payload = gemini_answer(
        plans=[
            plan("x265", 20.0, "slow", tune=""),
            plan("svt-av1", 27.0, tune="grain"),
        ]
    )

    advice = decided(payload)

    # Grain tuning is a judgement call, so clearing it is legitimate.
    assert advice.plans[1].tune is None
    # SVT-AV1's tune is a number in the parameter string, not a named bundle.
    assert advice.plans[0].tune is None
    assert any("as a parameter" in flag for flag in advice.rejected_flags)


def test_parameter_values_of_other_json_types_are_coerced() -> None:
    payload = gemini_answer(
        plans=[
            plan(
                "x265",
                20.0,
                "slow",
                params=[
                    {"name": "sao", "value": False},
                    {"name": "rc-lookahead", "value": 60},
                    {"name": "aq-strength", "value": 0.9},
                    {"name": "", "value": "1"},
                    {"name": "psy-rdoq", "value": None},
                ],
            )
        ]
    )

    params = decided(payload).plans[0].params

    assert params["sao"] == "0"
    assert params["rc-lookahead"] == "60"
    assert params["aq-strength"] == "0.9"
    assert "psy-rdoq" not in params


def test_prose_is_cleaned_before_it_reaches_the_page() -> None:
    payload = gemini_answer(
        summary="  Grain\x00 is\n\n the   point  ",
        notes=["x" * 500, "", "   ", 42, "A real note."],
    )

    advice = decided(payload)

    assert advice.summary == "Grain is the point"
    assert advice.notes[-1] == "A real note."
    assert advice.notes[0].endswith("…")
    assert len(advice.notes[0]) == 400


def test_the_number_of_lines_is_capped() -> None:
    payload = gemini_answer(notes=[f"Note {index}." for index in range(MAX_NOTES + 10)])

    assert len(decided(payload).notes) == MAX_NOTES


# --- the call itself ---------------------------------------------------------


def test_without_a_key_nothing_is_decided_and_nothing_is_blamed() -> None:
    client = GeminiClient(make_settings(), client=mock_client(lambda request: httpx2.Response(500)))

    decision = run(client.decide(request_for()))

    assert not client.enabled
    # Empty on both counts: it was never asked, so there is no problem to report.
    assert decision.advice is None and decision.problem is None


def test_the_key_travels_in_a_header_and_never_in_the_url() -> None:
    client, seen = answering(gemini_answer())

    run(client.decide(request_for()))

    assert seen[0].headers["x-goog-api-key"] == API_KEY
    # ?key= would put the key in every proxy and access log on the way out.
    assert API_KEY not in str(seen[0].url)
    assert "key=" not in str(seen[0].url)
    assert str(seen[0].url).endswith("/models/gemini-3.7-flash:generateContent")


def test_the_request_asks_for_json_against_the_schema() -> None:
    request = request_for()
    client, seen = answering(gemini_answer())

    run(client.decide(request))
    body = json.loads(seen[0].content)

    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["responseSchema"] == RESPONSE_SCHEMA
    assert body["generationConfig"]["temperature"] == 0.2
    assert body["systemInstruction"]["parts"][0]["text"] == SYSTEM_PROMPT
    assert json.loads(body["contents"][0]["parts"][0]["text"]) == build_context(request)


def test_a_decision_comes_back_as_the_advice() -> None:
    client, _ = answering(gemini_answer())

    decision = run(client.decide(request_for()))

    assert decision.problem is None
    assert decision.advice is not None
    assert decision.advice.source is AdviceSource.GEMINI
    assert decision.advice.plans[0].crf == 26.0


def test_the_same_question_is_only_asked_once() -> None:
    request = request_for()
    client, seen = answering(gemini_answer())

    first = run(client.decide(request))
    second = run(client.decide(request))

    assert len(seen) == 1
    # The payload is cached, not the Advice: commands are attached to plans later, so
    # two callers must not share one object.
    assert first.advice is not None and second.advice is not None
    assert first.advice is not second.advice
    assert first.advice.plans[0].crf == second.advice.plans[0].crf


def test_a_different_request_is_a_different_question() -> None:
    client, seen = answering(gemini_answer())

    run(client.decide(request_for()))
    run(client.decide(request_for(source=media(video=video_track(bitrate_bps=4_000_000)))))

    assert len(seen) == 2


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (400, "Gemini rejected the request (400)"),
        (403, "Gemini rejected the request (403)"),
        (404, "Gemini has no model named gemini-3.7-flash (404)."),
        (429, "Gemini rate limit reached"),
        (500, "Gemini returned HTTP 500."),
    ],
)
def test_an_http_failure_is_reported_and_decides_nothing(status_code: int, expected: str) -> None:
    handler, _ = recording_handler(httpx2.Response(status_code, json={"error": {}}))

    decision = run(gemini(handler).decide(request_for()))

    # No advice at all rather than a half-decided one: the pipeline puts the labelled
    # tables up instead, with this sentence over them.
    assert decision.advice is None
    assert decision.problem is not None and expected in decision.problem


def test_a_transport_failure_names_the_kind() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route")

    decision = run(gemini(handler).decide(request_for()))

    assert decision.problem == "Could not reach Gemini. (ConnectError)"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "declined to answer (SAFETY)"),
        ({"candidates": []}, "returned no candidates"),
        ({}, "returned no candidates"),
    ],
)
def test_a_refusal_or_an_empty_envelope_is_reported(body: dict[str, Any], expected: str) -> None:
    handler, _ = recording_handler(httpx2.Response(200, json=body))

    decision = run(gemini(handler).decide(request_for()))

    assert decision.problem is not None and expected in decision.problem


def test_a_truncated_answer_points_at_the_token_limit() -> None:
    handler, _ = recording_handler(
        httpx2.Response(200, json=envelope('{"summary": "half a sen', finishReason="MAX_TOKENS"))
    )

    decision = run(gemini(handler).decide(request_for()))

    assert decision.problem is not None and "GEMINI_MAX_OUTPUT_TOKENS" in decision.problem


def test_stopping_early_for_another_reason_is_reported_as_such() -> None:
    handler, _ = recording_handler(
        httpx2.Response(200, json=envelope(gemini_answer(), finishReason="RECITATION"))
    )

    decision = run(gemini(handler).decide(request_for()))

    assert decision.problem == "Gemini stopped early (RECITATION)."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "returned an empty answer"),
        ("   ", "returned an empty answer"),
        ("Here is my answer:", "was not the JSON it was asked for"),
        ("[1, 2, 3]", "was not a JSON object"),
    ],
)
def test_an_answer_that_is_not_the_requested_object_is_reported(text: str, expected: str) -> None:
    handler, _ = recording_handler(httpx2.Response(200, json=envelope(text)))

    decision = run(gemini(handler).decide(request_for()))

    assert decision.problem is not None and expected in decision.problem


def test_a_body_that_is_not_json_at_all_is_reported() -> None:
    handler, _ = recording_handler(httpx2.Response(200, text="<html>504</html>"))

    decision = run(gemini(handler).decide(request_for()))

    assert decision.problem is not None and "was not JSON" in decision.problem


def test_thinking_parts_are_skipped_and_text_parts_joined() -> None:
    answer = json.dumps(gemini_answer())
    split = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True},
                        {"text": answer[:20]},
                        {"functionCall": {"name": "nothing"}},
                        {"text": answer[20:]},
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }
    handler, _ = recording_handler(httpx2.Response(200, json=split))

    decision = run(gemini(handler).decide(request_for()))

    assert decision.advice is not None
    assert decision.advice.source is AdviceSource.GEMINI
    assert decision.advice.plans[0].crf == 26.0


# --- the technical-specs look-up ---------------------------------------------

SPECS_ANSWER: dict[str, Any] = {
    "negative_formats": ["35 mm"],
    "cinematographic_processes": ["Super 35"],
    "aspect_ratios": ["2.20 : 1"],
    "sound_mixes": ["Dolby Stereo", "70 mm 6-Track"],
    "confidence": "high",
}


def looking_up(answer: Any, **overrides: Any) -> tuple[GeminiClient, list[httpx2.Request]]:
    handler, seen = recording_handler(httpx2.Response(200, json=envelope(answer)))
    return gemini(handler, **overrides), seen


def test_the_lookup_asks_for_the_page_by_id_and_by_name() -> None:
    client, seen = looking_up(SPECS_ANSWER)

    run(client.technical_specs("tt0083658", title="Blade Runner", year=1982))

    asked = json.loads(seen[0].content)["contents"][0]["parts"][0]["text"]
    assert "tt0083658" in asked
    assert "Blade Runner (1982)" in asked
    # The URL is in the question so a grounded model can go and read the real page.
    assert "https://www.imdb.com/title/tt0083658/technical/" in asked


def test_the_rows_come_back_as_specs() -> None:
    client, _ = looking_up(SPECS_ANSWER)

    lookup = run(client.technical_specs("tt0083658"))

    assert lookup.specs.negative_formats == ["35 mm"]
    assert lookup.specs.sound_mixes == ["Dolby Stereo", "70 mm 6-Track"]
    assert lookup.confidence == "high"
    assert "confidence high" in lookup.caveat
    assert "not read from IMDb" in lookup.caveat


def test_a_grounded_lookup_searches_and_is_parsed_leniently() -> None:
    client, seen = looking_up(f"Here are the rows:\n```json\n{json.dumps(SPECS_ANSWER)}\n```\n")

    lookup = run(client.technical_specs("tt0083658"))

    body = json.loads(seen[0].content)
    assert body["tools"] == [{"google_search": {}}]
    # Grounding and structured output cannot be asked for at the same time.
    assert "responseSchema" not in body["generationConfig"]
    assert "single JSON object" in body["systemInstruction"]["parts"][0]["text"]
    assert lookup.grounded
    assert "searching the web" in lookup.caveat
    assert lookup.summary == "Gemini web search · confidence high"
    assert lookup.specs.negative_formats == ["35 mm"]


def test_without_grounding_the_lookup_is_held_to_the_schema() -> None:
    client, seen = looking_up(SPECS_ANSWER, gemini_web_grounding=False)

    lookup = run(client.technical_specs("tt0083658"))

    body = json.loads(seen[0].content)
    assert "tools" not in body
    assert body["generationConfig"]["responseSchema"] == TECHNICAL_SCHEMA
    assert body["systemInstruction"]["parts"][0]["text"] == TECHNICAL_PROMPT
    # A look-up is recall, not judgement: nothing about it wants variation.
    assert body["generationConfig"]["temperature"] == 0.0
    assert not lookup.grounded
    assert "recalled from training data" in lookup.caveat
    # The same fact in a few words, for a list row that has no room for the sentence.
    assert lookup.summary == "Gemini recall · confidence high"


def test_a_model_without_grounding_is_asked_again_without_it() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx2.Response(400, json={"error": {"message": "unsupported tool"}})
        return httpx2.Response(200, json=envelope(SPECS_ANSWER))

    lookup = run(gemini(handler).technical_specs("tt0083658"))

    assert len(seen) == 2
    assert "tools" not in json.loads(seen[1].content)
    assert lookup.specs.negative_formats == ["35 mm"]
    # The rows are worth less without a search behind them, and say so.
    assert not lookup.grounded


def test_a_spent_grounding_quota_is_asked_again_without_the_search() -> None:
    """The 429 that only ever hits the lookup: grounding is metered on its own.

    A key with plenty of ordinary generateContent left — the plan review goes through
    fine — can still be out of search requests, and rows from recall beat no rows.
    """
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if b"google_search" in request.content:
            return httpx2.Response(429, json={"error": {"message": "quota exceeded"}})
        return httpx2.Response(200, json=envelope(SPECS_ANSWER))

    lookup = run(gemini(handler).technical_specs("tt0083658"))

    assert len(seen) == 2
    assert "tools" not in json.loads(seen[1].content)
    assert lookup.specs.negative_formats == ["35 mm"]
    assert not lookup.grounded
    assert "recalled from training data" in lookup.caveat


def test_a_rate_limit_with_grounding_already_off_is_not_retried() -> None:
    handler, seen = recording_handler(httpx2.Response(429, json={"error": {}}))

    with pytest.raises(ProviderUnavailable, match="rate limit reached"):
        run(gemini(handler, gemini_web_grounding=False).technical_specs("tt0083658"))

    assert len(seen) == 1


def test_the_quota_that_ran_out_is_named_when_gemini_names_it() -> None:
    """ "Rate limit reached" alone reads as though the whole key were spent."""
    body = {
        "error": {
            "message": "You exceeded your current quota.",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "GroundingRequestsPerDayPerProject-FreeTier"}],
                }
            ],
        }
    }
    handler, _ = recording_handler(httpx2.Response(429, json=body))

    with pytest.raises(ProviderUnavailable) as raised:
        run(gemini(handler, gemini_web_grounding=False).technical_specs("tt0083658"))

    assert raised.value.detail == "GroundingRequestsPerDayPerProject-FreeTier"


def test_a_quota_with_no_violations_falls_back_to_the_message() -> None:
    handler, _ = recording_handler(
        httpx2.Response(429, json={"error": {"message": "Too many requests."}})
    )

    with pytest.raises(ProviderUnavailable) as raised:
        run(gemini(handler, gemini_web_grounding=False).technical_specs("tt0083658"))

    assert raised.value.detail == "Too many requests."


def test_a_refusal_that_is_not_about_grounding_is_only_tried_twice() -> None:
    handler, seen = recording_handler(httpx2.Response(400, json={"error": {}}))

    with pytest.raises(ProviderUnavailable, match="Gemini rejected the request"):
        run(gemini(handler).technical_specs("tt0083658"))

    assert len(seen) == 2


def test_a_lookup_that_knows_nothing_is_empty_rather_than_invented() -> None:
    client, _ = looking_up({"confidence": "low"})

    lookup = run(client.technical_specs("tt0083658"))

    assert lookup.specs.is_empty
    assert lookup.confidence == "low"


def test_rows_of_the_wrong_shape_are_dropped_not_trusted() -> None:
    client, _ = looking_up(
        {
            "negative_formats": ["16 mm"],
            "aspect_ratios": "1.37 : 1",
            "resolution": ["4K"],
            "cameras": [None, 12, "Bolex H16"],
            "confidence": "very sure indeed",
        }
    )

    lookup = run(client.technical_specs("tt0083658"))

    assert lookup.specs.negative_formats == ["16 mm"]
    assert lookup.specs.aspect_ratios == []
    assert lookup.specs.cameras == ["Bolex H16"]
    # An unrecognised confidence is the lowest one, never the benefit of the doubt.
    assert lookup.confidence == "low"


def test_the_same_film_is_only_looked_up_once() -> None:
    client, seen = looking_up(SPECS_ANSWER)

    first = run(client.technical_specs("tt0083658", title="Blade Runner", year=1982))
    second = run(client.technical_specs("tt0083658", title="Blade Runner", year=1982))

    assert len(seen) == 1
    assert first.specs == second.specs


def test_a_different_film_is_a_different_question() -> None:
    client, seen = looking_up(SPECS_ANSWER)

    run(client.technical_specs("tt0083658"))
    run(client.technical_specs("tt12042730"))

    assert len(seen) == 2


def test_a_lookup_without_a_key_names_the_paste_box() -> None:
    client = GeminiClient(make_settings(), client=mock_client(lambda request: httpx2.Response(500)))

    with pytest.raises(ProviderDisabled, match="Paste the technical page instead"):
        run(client.technical_specs("tt0083658"))


def test_a_failed_lookup_raises_rather_than_answering_emptily() -> None:
    handler, _ = recording_handler(httpx2.Response(500, json={"error": {}}))

    with pytest.raises(ProviderUnavailable, match=r"Gemini returned HTTP 500\."):
        run(gemini(handler).technical_specs("tt0083658"))


def test_the_lookup_schema_avoids_what_gemini_rejects() -> None:
    def walk(node: object) -> None:
        if isinstance(node, dict):
            assert "additionalProperties" not in node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(TECHNICAL_SCHEMA)
    assert set(TECHNICAL_SCHEMA["properties"]) == {*TechnicalSpecs.model_fields, "confidence"}


def test_the_lookup_prompt_forbids_filling_rows_out() -> None:
    # The whole risk of this feature is a model inventing a negative format that then
    # decides how much grain the encode keeps.
    assert "An empty list is the right answer for a row you do not know" in TECHNICAL_PROMPT
    assert "verbatim" in TECHNICAL_PROMPT
    assert "Never invent" in TECHNICAL_PROMPT
    assert "Do not inflate it" in TECHNICAL_PROMPT


def test_the_advice_context_says_where_the_rows_came_from() -> None:
    """Recalled rows and pasted rows are not worth the same, and the model is told which."""
    recalled = build_context(request_for(specs_source=SpecsSource.GEMINI))
    pasted = build_context(request_for(specs_source=SpecsSource.PASTED))

    assert "unverified" in recalled["imdb_technical_source"]
    assert "authoritative" in pasted["imdb_technical_source"]


def test_rows_with_no_provenance_claim_none() -> None:
    request = request_for()

    assert request.specs_source is None
    assert "imdb_technical_source" not in build_context(request)
