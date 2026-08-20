"""Grain inference — the premise of the app, so the rules get individual attention."""

from __future__ import annotations

import pytest

from app.advice.grain import infer_grain
from app.models import GrainLevel, SourceMedia, TechnicalSpecs
from tests.support import media, specs, video_track


@pytest.mark.parametrize(
    ("negative", "level", "origin"),
    [
        ("8 mm", GrainLevel.EXTREME, "8 mm"),
        ("Super 8", GrainLevel.EXTREME, "8 mm"),
        ("Super 16", GrainLevel.HEAVY, "Super 16 mm"),
        ("16 mm", GrainLevel.HEAVY, "16 mm"),
        ("35 mm (Techniscope)", GrainLevel.HEAVY, "2-perf 35 mm"),
        ("35 mm (3-perf)", GrainLevel.MODERATE, "Super 35 mm"),
        ("Super 35", GrainLevel.MODERATE, "Super 35 mm"),
        ("35 mm", GrainLevel.MODERATE, "35 mm"),
        ("65 mm", GrainLevel.LIGHT, "65 mm / large format"),
        ("IMAX 15/70", GrainLevel.LIGHT, "65 mm / large format"),
        ("Codex Digital", GrainLevel.NONE, "digital"),
    ],
)
def test_negative_format_sets_the_level(negative: str, level: GrainLevel, origin: str) -> None:
    profile = infer_grain(TechnicalSpecs(negative_formats=[negative]), SourceMedia())

    assert profile.level is level
    assert profile.origin_format == origin
    # A negative-format row is the strongest evidence there is.
    assert profile.confidence == pytest.approx(0.9)
    assert profile.reasons


def test_more_specific_formats_win_over_the_names_they_contain() -> None:
    # "Super 16" contains "16 mm"'s sibling and "65 mm" appears inside blow-up notes;
    # both would land on the wrong rule if the table were ordered loosely.
    assert (
        infer_grain(TechnicalSpecs(negative_formats=["Super 16"]), SourceMedia()).origin_format
        == "Super 16 mm"
    )
    blow_up = infer_grain(
        TechnicalSpecs(negative_formats=["65 mm (blown up from 35 mm)"]), SourceMedia()
    )
    # Both formats match this one entry, and the grainier of the two is what you see
    # in the print — but it is still one format, so no mixed-format penalty.
    assert blow_up.level is GrainLevel.MODERATE
    assert blow_up.confidence == pytest.approx(0.9)
    assert not any("mixed-format" in reason for reason in blow_up.reasons)


def test_one_entry_naming_two_formats_is_not_a_mixed_format_shoot() -> None:
    # "35 mm (Techniscope)" is a single negative described precisely. Reading it as
    # two formats would both dock confidence and claim grain "varies between shots".
    precise = infer_grain(TechnicalSpecs(negative_formats=["35 mm (Techniscope)"]), SourceMedia())
    assert precise.level is GrainLevel.HEAVY
    assert precise.origin_format == "2-perf 35 mm"
    assert precise.confidence == pytest.approx(0.9)
    assert len(precise.reasons) == 1

    # Two entries naming the same two formats genuinely is one.
    mixed = infer_grain(TechnicalSpecs(negative_formats=["35 mm", "Techniscope"]), SourceMedia())
    assert mixed.confidence == pytest.approx(0.8)
    assert any("mixed-format" in reason for reason in mixed.reasons)


def test_the_grainiest_of_several_formats_dominates() -> None:
    profile = infer_grain(
        TechnicalSpecs(negative_formats=["35 mm", "16 mm"], cinematographic_processes=["Super 35"]),
        SourceMedia(),
    )

    assert profile.level is GrainLevel.HEAVY
    assert profile.origin_format == "16 mm"
    # Mixed-format shoots are less predictable, so confidence drops a notch.
    assert profile.confidence == pytest.approx(0.8)
    assert any("mixed-format" in reason for reason in profile.reasons)
    assert any("35 mm" in reason for reason in profile.reasons)


def test_several_formats_of_the_same_grain_level_are_not_a_mixed_shoot() -> None:
    # Blade Runner's own rows: 35 mm, Panavision anamorphic and Super 35 all describe
    # the same moderately grainy 35 mm look, so nothing "varies between shots".
    profile = infer_grain(specs(), SourceMedia())

    assert profile.level is GrainLevel.MODERATE
    # A tie on grain level goes to the more specific rule.
    assert profile.origin_format == "Super 35 mm"
    assert profile.confidence == pytest.approx(0.9)
    assert not any("mixed-format" in reason for reason in profile.reasons)


def test_a_process_row_alone_is_weaker_evidence_than_a_negative_row() -> None:
    profile = infer_grain(TechnicalSpecs(cinematographic_processes=["Super 35"]), SourceMedia())

    assert profile.level is GrainLevel.MODERATE
    assert profile.confidence == pytest.approx(0.7)


def test_digital_intermediate_is_a_finish_not_a_capture() -> None:
    # The trap: "Digital Intermediate (2K)" contains "Digital", which would otherwise
    # match the digital-capture rule and strip grain protection from a 35 mm film.
    profile = infer_grain(
        TechnicalSpecs(
            negative_formats=["35 mm"],
            cinematographic_processes=["Digital Intermediate (2K)", "Super 35"],
        ),
        SourceMedia(),
    )

    assert profile.level is GrainLevel.MODERATE
    assert any("signed off in the DI" in reason for reason in profile.reasons)


def test_a_film_camera_without_a_negative_row_assumes_35_mm() -> None:
    profile = infer_grain(
        TechnicalSpecs(cameras=["Arriflex 435, Zeiss Ultra Prime Lenses"]), SourceMedia()
    )

    assert profile.level is GrainLevel.MODERATE
    assert profile.origin_format == "35 mm"
    assert profile.confidence == pytest.approx(0.6)
    assert any("film camera but no negative format" in reason for reason in profile.reasons)


@pytest.mark.parametrize("year", [2012, 2020, 2026])
def test_a_modern_year_with_no_imdb_data_assumes_digital(year: int) -> None:
    profile = infer_grain(TechnicalSpecs(), SourceMedia(), year=year)

    assert profile.level is GrainLevel.NONE
    assert profile.origin_format == "digital"
    assert profile.confidence == pytest.approx(0.4)
    assert str(year) in " ".join(profile.reasons)


@pytest.mark.parametrize("year", [1927, 1982, 2000])
def test_an_old_year_with_no_imdb_data_assumes_35_mm(year: int) -> None:
    profile = infer_grain(TechnicalSpecs(), SourceMedia(), year=year)

    assert profile.level is GrainLevel.MODERATE
    assert profile.origin_format == "35 mm"
    assert profile.confidence == pytest.approx(0.4)


@pytest.mark.parametrize("year", [2001, 2007, 2011])
def test_the_transition_years_admit_they_decide_nothing(year: int) -> None:
    profile = infer_grain(TechnicalSpecs(), SourceMedia(), year=year)

    assert profile.level is GrainLevel.LIGHT
    assert profile.origin_format is None
    assert profile.confidence == pytest.approx(0.25)
    assert any("settles nothing" in reason for reason in profile.reasons)


def test_nothing_at_all_is_cautious_and_says_so() -> None:
    profile = infer_grain(TechnicalSpecs(), SourceMedia())

    assert profile.level is GrainLevel.LIGHT
    assert profile.confidence == pytest.approx(0.25)
    assert any("Nothing to go on" in reason for reason in profile.reasons)
    assert not profile.user_override


def test_a_starved_source_lowers_confidence_in_grainy_films() -> None:
    # 1.5 Mb/s over 1080p24: a web rip whose grain a previous encoder already ate.
    starved = media(video=video_track(bitrate_bps=1_500_000))
    assert starved.video is not None
    assert starved.video.bits_per_pixel is not None
    assert starved.video.bits_per_pixel < 0.05

    profile = infer_grain(specs(), starved, year=1982)

    assert profile.level is GrainLevel.MODERATE
    # 0.9 for the negative-format match, less 0.15 for the smoothed-over source.
    assert profile.confidence == pytest.approx(0.75)
    assert any("bits per pixel" in reason for reason in profile.reasons)


def test_a_healthy_source_does_not_lower_confidence() -> None:
    profile = infer_grain(TechnicalSpecs(negative_formats=["35 mm"]), media())

    assert profile.confidence == pytest.approx(0.9)
    assert not any("bits per pixel" in reason for reason in profile.reasons)


def test_a_starved_digital_source_is_not_penalised() -> None:
    # There was no grain to lose, so the bitrate says nothing about confidence.
    profile = infer_grain(
        TechnicalSpecs(negative_formats=["Arri Alexa"]),
        media(video=video_track(bitrate_bps=1_500_000)),
    )

    assert profile.level is GrainLevel.NONE
    assert profile.confidence == pytest.approx(0.9)


def test_confidence_never_leaves_the_unit_interval() -> None:
    # Two penalties on top of the weaker 0.7 base would otherwise undershoot 0.1.
    profile = infer_grain(
        TechnicalSpecs(cinematographic_processes=["Super 35", "16 mm", "Techniscope"]),
        media(video=video_track(bitrate_bps=100_000)),
    )

    assert 0.1 <= profile.confidence <= 1.0


def test_an_override_wins_over_every_rule() -> None:
    profile = infer_grain(
        TechnicalSpecs(negative_formats=["Arri Alexa"]),
        media(),
        year=2020,
        override=GrainLevel.EXTREME,
    )

    assert profile.level is GrainLevel.EXTREME
    assert profile.confidence == 1.0
    assert profile.user_override
    assert profile.origin_format is None
    assert profile.reasons == ["You set the grain level to extreme by hand."]
