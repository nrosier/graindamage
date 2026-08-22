"""Gemini as a reviewer of the baseline, and never as a single point of failure.

Three separable things are tested here: what the model is *told* (``build_context``),
what happens to what it *says* (``merge_advice``), and what happens when the call
itself goes wrong (``GeminiClient.annotate``, which must never raise).

The technical-specs look-up is the one call whose failure the user does see, because
without rows there is nothing to show in the panel — so it raises rather than pretends,
and everything it does return is labelled with where it came from.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

from app.advice.rules import REAL_GRAIN_CRF_CEILING, build_advice
from app.models import (
    Advice,
    AdviceSource,
    Encoder,
    EncodeRequest,
    SourceTool,
    SpecsSource,
    TechnicalSpecs,
)
from app.providers import ProviderDisabled, ProviderUnavailable
from app.providers.gemini import (
    MAX_NOTES,
    MAX_OVERVIEW_CHARS,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    TECHNICAL_PROMPT,
    TECHNICAL_SCHEMA,
    GeminiClient,
    build_context,
    merge_advice,
)
from app.sources import parse_source
from tests.support import (
    Handler,
    fixture,
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


def baseline(**overrides: Any) -> tuple[Advice, EncodeRequest]:
    request = request_for(**overrides)
    return build_advice(request), request


def model_payload(**overrides: Any) -> dict[str, Any]:
    """A well-formed answer of the shape the response schema asks for."""
    payload: dict[str, Any] = {
        "summary": "A 1982 anamorphic 35 mm negative with grain the DI kept on purpose.",
        "notes": ["The opening flyover is the hardest shot in the film for any encoder."],
        "warnings": [],
        "plans": [
            {
                "encoder": "svt-av1",
                "crf": 28.0,
                "preset": "3",
                "params": [{"name": "aq-mode", "value": "2"}],
                "rationale": ["A slower preset pays for itself on the smoke and rain."],
            },
            {
                "encoder": "x265",
                "crf": 19.0,
                "preset": "slower",
                "tune": "grain",
                "params": [{"name": "psy-rd", "value": "1.5"}],
                "rationale": [],
            },
        ],
    }
    return {**payload, **overrides}


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


# --- what the model is told --------------------------------------------------


def test_the_local_file_path_is_never_sent() -> None:
    advice, request = baseline(input_path="/home/nick/rips/Blade Runner (1982).mkv")

    body = json.dumps(build_context(request, advice))

    # The model has no use for a path on someone's NAS, and no business seeing it.
    assert "/home/nick" not in body
    assert "input_path" not in body
    assert "output_stem" not in body
    assert "blade-runner-1982-1080p" not in body


def test_the_api_key_is_never_in_the_context() -> None:
    advice, request = baseline()

    assert API_KEY not in json.dumps(build_context(request, advice))


def test_the_context_carries_the_film_and_the_baseline() -> None:
    advice, request = baseline()

    context = build_context(request, advice)

    assert context["film"]["title"] == "Blade Runner"
    assert context["film"]["year"] == 1982
    assert context["film"]["directors"] == ["Ridley Scott"]
    assert context["grain_estimate"]["level"] == "moderate"
    assert context["grain_estimate"]["origin_format"] == "Super 35 mm"
    assert context["grain_estimate"]["photochemical"] is True
    assert context["grain_estimate"]["set_by_user"] is False
    assert context["imdb_technical"]["negative_formats"] == ["35 mm"]
    assert context["preferences"] == {"speed": "balanced", "size": "balanced"}
    proposal = context["proposal"]
    assert [plan["encoder"] for plan in proposal["plans"]] == ["svt-av1", "x265"]
    assert proposal["plans"][0]["crf"] == advice.plans[0].crf
    assert proposal["plans"][0]["params"] == advice.plans[0].params
    # Named a proposal, and carrying the one number the model is asked to reason with.
    assert "has not read this film" in proposal["made_by"]
    assert proposal["svt_av1_real_grain_crf_ceiling"] == REAL_GRAIN_CRF_CEILING


def test_the_context_describes_the_source_file() -> None:
    advice, request = baseline(
        source=parse_source(fixture("ffprobe_uhd_hdr.json")).media, source_tool=SourceTool.FFPROBE
    )

    source = build_context(request, advice)["source"]

    assert source["resolution_class"] == "2160p"
    assert source["parsed_from"] == "ffprobe"
    assert source["video"]["codec"] == "hevc"
    assert source["video"]["bit_depth"] == 10
    assert source["video"]["hdr"] is True
    assert source["video"]["has_mastering_display"] is True
    assert source["video"]["max_cll"] == 1000
    # Four decimal places is plenty for a judgement about compression history.
    assert source["video"]["bits_per_pixel"] == pytest.approx(0.2923, abs=5e-5)
    assert [track["codec"] for track in source["audio"]] == ["truehd", "ac3"]
    assert source["subtitle_count"] == 1


def test_absent_facts_are_omitted_rather_than_sent_as_null() -> None:
    # No film, no technical rows, no video track: a null-strewn context invites the
    # model to fill the gaps in, which is exactly what it must not do here.
    advice, request = baseline(
        movie=None, specs=TechnicalSpecs(), source=media(video=None, audio=[])
    )

    context = build_context(request, advice)

    assert "film" not in context
    assert "imdb_technical" not in context
    assert "video" not in context["source"]
    assert "audio" not in context["source"]
    assert "release_year" not in context


def test_a_parsed_year_reaches_the_model_when_no_film_was_looked_up() -> None:
    # Without a TMDB key the CLI has only the year in the filename, and that year is
    # the whole input to the older-film half of the grain rules.
    advice, request = baseline(movie=None, fallback_year=1962)

    context = build_context(request, advice)

    assert "film" not in context
    assert context["release_year"] == 1962


def test_a_long_overview_is_truncated() -> None:
    advice, request = baseline(movie=movie(overview="x" * 2_000))

    assert len(build_context(request, advice)["film"]["overview"]) == MAX_OVERVIEW_CHARS


def test_at_most_eight_audio_tracks_are_described() -> None:
    # A dozen dubs is a real disc; describing all of them wastes the context window.
    advice, request = baseline(source=media(audio=media().audio * 12))

    assert len(build_context(request, advice)["source"]["audio"]) == 8


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
    assert RESPONSE_SCHEMA["required"] == ["summary", "plans"]


def test_the_system_prompt_states_the_limits_it_will_be_held_to() -> None:
    # Anything the validator silently enforces should be in the prompt too, or the
    # model spends its output on suggestions that get thrown away.
    assert "There is no limit on how far you may move it" in SYSTEM_PROMPT
    assert "10-55 for SVT-AV1" in SYSTEM_PROMPT  # the CRF range that is still enforced
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


# --- merging ----------------------------------------------------------------


def test_a_clean_answer_is_folded_into_the_baseline() -> None:
    advice, request = baseline()

    reviewed = merge_advice(advice, model_payload(), request)

    assert reviewed.source is AdviceSource.GEMINI
    assert reviewed.summary is not None and reviewed.summary.startswith("A 1982 anamorphic")
    assert reviewed.notes[-1].startswith("The opening flyover")
    assert reviewed.rejected_flags == []

    svt, x265 = reviewed.plans
    assert (svt.crf, svt.preset) == (28.0, "3")
    assert svt.params["aq-mode"] == "2"
    assert svt.rationale[-1].startswith("A slower preset")
    assert (x265.crf, x265.preset, x265.tune) == (19.0, "slower", "grain")
    assert x265.params["psy-rd"] == "1.5"


def test_the_baseline_is_left_intact_for_the_page_to_fall_back_to() -> None:
    advice, request = baseline()
    before = advice.model_dump()

    merge_advice(advice, model_payload(), request)

    assert advice.model_dump() == before


def test_a_changed_crf_is_shown_as_its_own_adjustment() -> None:
    advice, request = baseline()
    original = advice.plans[0].crf

    reviewed = merge_advice(advice, model_payload(), request)
    step = reviewed.plans[0].adjustments[-1]

    assert step.label == "Gemini"
    assert step.delta == pytest.approx(28.0 - original)
    # The CRF is arguable, so every point of it stays traceable.
    assert sum(adjustment.delta for adjustment in reviewed.plans[0].adjustments) == pytest.approx(
        28.0
    )


def test_a_changed_crf_re_estimates_the_bitrate() -> None:
    advice, request = baseline()
    was = advice.plans[0].estimated_bitrate_bps

    reviewed = merge_advice(advice, model_payload(), request)

    assert was is not None and reviewed.plans[0].estimated_bitrate_bps is not None
    assert reviewed.plans[0].estimated_bitrate_bps < was


def test_an_unchanged_crf_adds_no_adjustment() -> None:
    advice, request = baseline()
    payload = model_payload(
        plans=[{"encoder": "svt-av1", "crf": advice.plans[0].crf, "preset": "4", "rationale": []}]
    )

    reviewed = merge_advice(advice, payload, request)

    assert not any(step.label == "Gemini" for step in reviewed.plans[0].adjustments)


def test_a_crf_far_from_the_proposal_is_kept_and_shown_as_the_models_own() -> None:
    """The model has read the film; the proposal came from a table. It gets to disagree."""
    advice, request = baseline()
    payload = model_payload(
        plans=[{"encoder": "x265", "crf": 15.0, "preset": "slow", "rationale": []}]
    )

    reviewed = merge_advice(advice, payload, request)

    assert reviewed.plans[1].crf == 15.0
    assert reviewed.rejected_flags == []
    step = next(step for step in reviewed.plans[1].adjustments if step.label == "Gemini")
    assert step.delta == round(15.0 - advice.plans[1].crf, 1)


def test_a_crf_the_encoder_would_refuse_is_still_clamped() -> None:
    advice, request = baseline()
    payload = model_payload(plans=[{"encoder": "x265", "crf": 99.0, "preset": "slow"}])

    reviewed = merge_advice(advice, payload, request)

    assert reviewed.plans[1].crf == 40.0
    assert any("clamped to 40" in flag for flag in reviewed.rejected_flags)


def test_a_plan_that_starves_its_own_grain_is_reported_and_left_standing() -> None:
    """The coherence check reports; it does not rewrite. The model may have a reason."""
    advice, request = baseline()
    payload = model_payload(plans=[{"encoder": "svt-av1", "crf": 31.0, "preset": "4"}])

    reviewed = merge_advice(advice, payload, request)

    plan = reviewed.plans[0]
    # Untouched: the CRF the model asked for, over the baseline's film-grain-denoise=0.
    assert plan.crf == 31.0
    assert plan.params["film-grain-denoise"] == "0"
    complaint = reviewed.warnings[-1]
    assert "film-grain-denoise=0" in complaint
    assert f"above the {REAL_GRAIN_CRF_CEILING:g}" in complaint
    assert "why this film is an exception" in complaint


def test_a_high_crf_over_replaced_grain_is_coherent_and_says_nothing() -> None:
    advice, request = baseline()
    payload = model_payload(
        plans=[
            {
                "encoder": "svt-av1",
                "crf": 33.0,
                "preset": "4",
                "params": [
                    {"name": "film-grain-denoise", "value": "1"},
                    {"name": "film-grain", "value": "10"},
                ],
            }
        ]
    )

    reviewed = merge_advice(advice, payload, request)

    assert reviewed.plans[0].crf == 33.0
    assert reviewed.warnings == advice.warnings


def test_coded_grain_under_the_ceiling_says_nothing_either() -> None:
    advice, request = baseline()
    payload = model_payload(plans=[{"encoder": "svt-av1", "crf": 26.0, "preset": "4"}])

    reviewed = merge_advice(advice, payload, request)

    assert reviewed.plans[0].params["film-grain-denoise"] == "0"
    assert reviewed.warnings == advice.warnings


def test_the_sources_own_colour_metadata_always_wins() -> None:
    advice, request = baseline(source=parse_source(fixture("ffprobe_uhd_hdr.json")).media)
    real = advice.plans[1].params["master-display"]
    payload = model_payload(
        plans=[
            {
                "encoder": "x265",
                "crf": advice.plans[1].crf,
                "preset": "slow",
                "params": [
                    {"name": "master-display", "value": "G(1,1)B(1,1)R(1,1)WP(1,1)L(1,1)"},
                    {"name": "colorprim", "value": "bt709"},
                    {"name": "max-cll", "value": "4000,600"},
                ],
                "rationale": [],
            }
        ]
    )

    reviewed = merge_advice(advice, payload, request)

    # The model has not seen the file; a plausible-looking guess here would silently
    # mis-tag the HDR of the finished encode.
    assert reviewed.plans[1].params["master-display"] == real
    assert reviewed.plans[1].params["colorprim"] == "bt2020"
    assert reviewed.plans[1].params["max-cll"] == "1000,400"


def test_a_protected_parameter_the_source_does_not_justify_is_removed() -> None:
    # An SDR source has no mastering display, so an invented one is not "additional
    # metadata", it is a lie about the master.
    advice, request = baseline()
    assert "master-display" not in advice.plans[1].params
    payload = model_payload(
        plans=[
            {
                "encoder": "x265",
                "crf": advice.plans[1].crf,
                "preset": "slow",
                "params": [{"name": "master-display", "value": "G(1,1)B(1,1)R(1,1)WP(1,1)L(1,1)"}],
                "rationale": [],
            }
        ]
    )

    reviewed = merge_advice(advice, payload, request)

    assert "master-display" not in reviewed.plans[1].params


def test_baseline_parameters_the_model_ignored_are_kept() -> None:
    advice, request = baseline()
    payload = model_payload(
        plans=[{"encoder": "svt-av1", "crf": 27.0, "preset": "4", "params": [], "rationale": []}]
    )

    reviewed = merge_advice(advice, payload, request)

    # Omission is not deletion: losing keyint because a model forgot it would be a
    # far worse failure than being unable to drop it.
    assert reviewed.plans[0].params["keyint"] == advice.plans[0].params["keyint"]


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
    advice, request = baseline()
    target = advice.plan_for(expected)
    assert target is not None
    # A point off the baseline, so the drift limit is not what is under test here, and
    # the preset each encoder actually spells that way.
    crf = target.crf - 1.0
    preset = "4" if expected is Encoder.SVT_AV1 else "slower"
    payload = model_payload(plans=[{"encoder": alias, "crf": crf, "preset": preset}])

    reviewed = merge_advice(advice, payload, request)

    plan = reviewed.plan_for(expected)
    assert plan is not None and plan.crf == crf
    assert plan.preset == preset
    # Nothing rejected: the alias resolved rather than being read as a new encoder.
    assert reviewed.rejected_flags == []


def test_a_plan_for_an_encoder_that_does_not_exist_is_reported() -> None:
    advice, request = baseline()
    payload = model_payload(plans=[{"encoder": "vp9", "crf": 30.0, "preset": "4"}])

    reviewed = merge_advice(advice, payload, request)

    assert reviewed.rejected_flags == ["a plan for unknown encoder 'vp9'"]


def test_a_second_plan_for_the_same_encoder_is_ignored() -> None:
    advice, request = baseline()
    payload = model_payload(
        plans=[
            {"encoder": "x265", "crf": 21.0, "preset": "slow"},
            {"encoder": "x265", "crf": 16.0, "preset": "veryslow"},
        ]
    )

    reviewed = merge_advice(advice, payload, request)

    assert reviewed.plans[1].crf == 21.0
    assert reviewed.rejected_flags == ["a second x265 plan"]


def test_a_plan_the_baseline_does_not_have_is_reported() -> None:
    advice, request = baseline()
    only_av1 = advice.model_copy(deep=True)
    only_av1.plans = [plan for plan in only_av1.plans if plan.encoder is Encoder.SVT_AV1]

    reviewed = merge_advice(only_av1, model_payload(), request)

    assert reviewed.rejected_flags == ["a x265 plan the baseline does not have"]


def test_a_crf_that_is_not_a_number_is_reported_and_ignored() -> None:
    advice, request = baseline()
    payload = model_payload(plans=[{"encoder": "svt-av1", "crf": "twenty-eight", "preset": "4"}])

    reviewed = merge_advice(advice, payload, request)

    assert reviewed.plans[0].crf == advice.plans[0].crf
    assert reviewed.rejected_flags == ["CRF 'twenty-eight' (not a number)"]


def test_an_x265_tune_can_be_cleared_but_svt_av1_gets_none() -> None:
    advice, request = baseline()
    assert advice.plans[1].tune == "grain"
    payload = model_payload(
        plans=[
            {"encoder": "x265", "crf": 20.0, "preset": "slow", "tune": ""},
            {"encoder": "svt-av1", "crf": 27.0, "preset": "4", "tune": "grain"},
        ]
    )

    reviewed = merge_advice(advice, payload, request)

    # Grain tuning is a judgement call, so clearing it is legitimate.
    assert reviewed.plans[1].tune is None
    # SVT-AV1's tune is a number in the parameter string, not a named bundle.
    assert reviewed.plans[0].tune is None
    assert any("as a parameter" in flag for flag in reviewed.rejected_flags)


def test_parameter_values_of_other_json_types_are_coerced() -> None:
    advice, request = baseline()
    payload = model_payload(
        plans=[
            {
                "encoder": "x265",
                "crf": 20.0,
                "preset": "slow",
                "params": [
                    {"name": "sao", "value": False},
                    {"name": "rc-lookahead", "value": 60},
                    {"name": "aq-strength", "value": 0.9},
                    {"name": "", "value": "1"},
                    {"name": "psy-rdoq", "value": None},
                ],
            }
        ]
    )

    reviewed = merge_advice(advice, payload, request)

    assert reviewed.plans[1].params["sao"] == "0"
    assert reviewed.plans[1].params["rc-lookahead"] == "60"
    assert reviewed.plans[1].params["aq-strength"] == "0.9"
    assert "psy-rdoq" not in reviewed.plans[1].params


def test_prose_is_cleaned_before_it_reaches_the_page() -> None:
    advice, request = baseline()
    payload = model_payload(
        summary="  Grain\x00 is\n\n the   point  ",
        notes=["x" * 500, "", "   ", 42, "A real note."],
    )

    reviewed = merge_advice(advice, payload, request)

    assert reviewed.summary == "Grain is the point"
    assert reviewed.notes[-1] == "A real note."
    assert reviewed.notes[-2].endswith("…")
    assert len(reviewed.notes[-2]) == 400


def test_a_note_the_baseline_already_makes_is_not_repeated() -> None:
    advice, request = baseline()
    existing = advice.notes[0]

    reviewed = merge_advice(
        advice, model_payload(notes=[existing.upper(), "Something new."]), request
    )

    assert reviewed.notes.count(existing) == 1
    assert existing.upper() not in reviewed.notes
    assert reviewed.notes[-1] == "Something new."


def test_the_number_of_added_lines_is_capped() -> None:
    advice, request = baseline()
    payload = model_payload(notes=[f"Note {index}." for index in range(MAX_NOTES + 10)])

    reviewed = merge_advice(advice, payload, request)

    assert len(reviewed.notes) == len(advice.notes) + MAX_NOTES


def test_an_answer_with_nothing_in_it_leaves_the_baseline_alone() -> None:
    advice, request = baseline()

    reviewed = merge_advice(advice, {}, request)

    assert reviewed.summary == advice.summary
    assert reviewed.plans[0].crf == advice.plans[0].crf
    assert reviewed.rejected_flags == []
    # It is still a reviewed answer: the page says so, and nothing was changed.
    assert reviewed.source is AdviceSource.GEMINI


# --- the call itself ---------------------------------------------------------


def test_without_a_key_the_baseline_is_returned_as_it_stands() -> None:
    advice, request = baseline()
    client = GeminiClient(make_settings(), client=mock_client(lambda request: httpx2.Response(500)))

    assert not client.enabled
    assert run(client.annotate(request, advice)) is advice


def test_the_key_travels_in_a_header_and_never_in_the_url() -> None:
    advice, request = baseline()
    client, seen = answering(model_payload())

    run(client.annotate(request, advice))

    assert seen[0].headers["x-goog-api-key"] == API_KEY
    # ?key= would put the key in every proxy and access log on the way out.
    assert API_KEY not in str(seen[0].url)
    assert "key=" not in str(seen[0].url)
    assert str(seen[0].url).endswith("/models/gemini-3.7-flash:generateContent")


def test_the_request_asks_for_json_against_the_schema() -> None:
    advice, request = baseline()
    client, seen = answering(model_payload())

    run(client.annotate(request, advice))
    body = json.loads(seen[0].content)

    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["responseSchema"] == RESPONSE_SCHEMA
    assert body["generationConfig"]["temperature"] == 0.2
    assert body["systemInstruction"]["parts"][0]["text"] == SYSTEM_PROMPT
    assert json.loads(body["contents"][0]["parts"][0]["text"]) == build_context(request, advice)


def test_a_reviewed_answer_comes_back_merged() -> None:
    advice, request = baseline()
    client, _ = answering(model_payload())

    reviewed = run(client.annotate(request, advice))

    assert reviewed.source is AdviceSource.GEMINI
    assert reviewed.plans[0].crf == 28.0


def test_the_same_question_is_only_asked_once() -> None:
    advice, request = baseline()
    client, seen = answering(model_payload())

    first = run(client.annotate(request, advice))
    second = run(client.annotate(request, advice))

    assert len(seen) == 1
    # The payload is cached, not the Advice: commands are attached to plans later, so
    # two callers must not share one object.
    assert first is not second
    assert first.plans[0].crf == second.plans[0].crf


def test_a_different_request_is_a_different_question() -> None:
    advice, request = baseline()
    other_advice, other_request = baseline(source=media(video=video_track(bitrate_bps=4_000_000)))
    client, seen = answering(model_payload())

    run(client.annotate(request, advice))
    run(client.annotate(other_request, other_advice))

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
def test_an_http_failure_costs_a_warning_and_nothing_else(status_code: int, expected: str) -> None:
    advice, request = baseline()
    handler, _ = recording_handler(httpx2.Response(status_code, json={"error": {}}))

    reviewed = run(gemini(handler).annotate(request, advice))

    assert expected in reviewed.warnings[-1]
    assert "deterministic baseline" in reviewed.warnings[-1]
    # The advice is complete without Gemini, and says so rather than pretending.
    assert reviewed.source is AdviceSource.BASELINE
    assert reviewed.plans[0].crf == advice.plans[0].crf
    assert reviewed.plans[0].params == advice.plans[0].params


def test_a_transport_failure_names_the_kind() -> None:
    advice, request = baseline()

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route")

    reviewed = run(gemini(handler).annotate(request, advice))

    assert "Could not reach Gemini. (ConnectError)" in reviewed.warnings[-1]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "declined to answer (SAFETY)"),
        ({"candidates": []}, "returned no candidates"),
        ({}, "returned no candidates"),
    ],
)
def test_a_refusal_or_an_empty_envelope_is_reported(body: dict[str, Any], expected: str) -> None:
    advice, request = baseline()
    handler, _ = recording_handler(httpx2.Response(200, json=body))

    reviewed = run(gemini(handler).annotate(request, advice))

    assert expected in reviewed.warnings[-1]


def test_a_truncated_answer_points_at_the_token_limit() -> None:
    advice, request = baseline()
    handler, _ = recording_handler(
        httpx2.Response(200, json=envelope('{"summary": "half a sen', finishReason="MAX_TOKENS"))
    )

    reviewed = run(gemini(handler).annotate(request, advice))

    assert "GEMINI_MAX_OUTPUT_TOKENS" in reviewed.warnings[-1]


def test_stopping_early_for_another_reason_is_reported_as_such() -> None:
    advice, request = baseline()
    handler, _ = recording_handler(
        httpx2.Response(200, json=envelope(model_payload(), finishReason="RECITATION"))
    )

    reviewed = run(gemini(handler).annotate(request, advice))

    assert "Gemini stopped early (RECITATION)." in reviewed.warnings[-1]


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
    advice, request = baseline()
    handler, _ = recording_handler(httpx2.Response(200, json=envelope(text)))

    reviewed = run(gemini(handler).annotate(request, advice))

    assert expected in reviewed.warnings[-1]


def test_a_body_that_is_not_json_at_all_is_reported() -> None:
    advice, request = baseline()
    handler, _ = recording_handler(httpx2.Response(200, text="<html>504</html>"))

    reviewed = run(gemini(handler).annotate(request, advice))

    assert "was not JSON" in reviewed.warnings[-1]


def test_thinking_parts_are_skipped_and_text_parts_joined() -> None:
    advice, request = baseline()
    answer = json.dumps(model_payload())
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

    reviewed = run(gemini(handler).annotate(request, advice))

    assert reviewed.source is AdviceSource.GEMINI
    assert reviewed.plans[0].crf == 28.0


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
    advice, request = baseline()

    request.specs_source = SpecsSource.GEMINI
    recalled = build_context(request, advice)
    request.specs_source = SpecsSource.PASTED
    pasted = build_context(request, advice)

    assert "unverified" in recalled["imdb_technical_source"]
    assert "authoritative" in pasted["imdb_technical_source"]


def test_rows_with_no_provenance_claim_none() -> None:
    advice, request = baseline()

    assert request.specs_source is None
    assert "imdb_technical_source" not in build_context(request, advice)
