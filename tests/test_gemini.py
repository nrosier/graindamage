"""Gemini as a reviewer of the baseline, and never as a single point of failure.

Three separable things are tested here: what the model is *told* (``build_context``),
what happens to what it *says* (``merge_advice``), and what happens when the call
itself goes wrong (``GeminiClient.annotate``, which must never raise).
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

from app.advice.rules import build_advice
from app.advice.validate import MAX_CRF_DRIFT
from app.models import Advice, AdviceSource, Encoder, EncodeRequest, SourceTool, TechnicalSpecs
from app.providers.gemini import (
    MAX_NOTES,
    MAX_OVERVIEW_CHARS,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
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
    assert context["grain_estimate"]["set_by_user"] is False
    assert context["imdb_technical"]["negative_formats"] == ["35 mm"]
    assert context["preferences"] == {"speed": "balanced", "size": "balanced"}
    assert [plan["encoder"] for plan in context["baseline_plans"]] == ["svt-av1", "x265"]
    assert context["baseline_plans"][0]["crf"] == advice.plans[0].crf
    assert context["baseline_plans"][0]["params"] == advice.plans[0].params


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
    assert f"within {MAX_CRF_DRIFT:g} points" in SYSTEM_PROMPT
    assert "slower" in SYSTEM_PROMPT  # the x265 preset ladder
    assert "film-grain" in SYSTEM_PROMPT  # the SVT-AV1 parameter allowlist
    assert "will be overwritten with the file's own values" in SYSTEM_PROMPT


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


def test_a_crf_beyond_the_drift_limit_is_clamped_and_declared() -> None:
    advice, request = baseline()
    payload = model_payload(
        plans=[{"encoder": "x265", "crf": 34.0, "preset": "slow", "rationale": []}]
    )

    reviewed = merge_advice(advice, payload, request)

    assert reviewed.plans[1].crf == advice.plans[1].crf + MAX_CRF_DRIFT
    assert any("pulled back" in flag for flag in reviewed.rejected_flags)


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
