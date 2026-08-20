"""Infer how grainy a film actually is, from how it was shot.

This is the whole premise of the app: grain is the single biggest factor in how a
film should be encoded, and it is a property of the *negative*, not of the file. A
Super 16 blow-up and an Alexa capture at the same resolution and bitrate want
materially different settings, and no amount of looking at the file tells you which
you have — but IMDb's ``Negative Format`` row does.

Grain size scales inversely with negative area, so the ordering falls out of the
formats themselves: 8 mm is worse than 16 mm is worse than 2-perf 35 mm is worse than
4-perf 35 mm is worse than 65 mm, and digital acquisition has no photochemical grain
at all. Everything below is that ordering, plus the vocabulary IMDb uses to describe
each format.

When IMDb tells us nothing, the release year is a weak last resort (digital
acquisition became the norm for mainstream production around 2010–2012) and the
resulting profile says so through a low confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import GrainLevel, GrainProfile, SourceMedia, TechnicalSpecs


@dataclass(frozen=True, slots=True)
class _FormatRule:
    pattern: str
    level: GrainLevel
    origin: str
    reason: str


# Ordered most specific first: "Super 16" must win over the bare "16 mm" it contains,
# and "65 mm" must be tested before the "35 mm" in "blown up from 35 mm".
_FORMAT_RULES: tuple[_FormatRule, ...] = (
    _FormatRule(
        r"\b(?:8\s*mm|super\s*8|regular\s*8|standard\s*8)\b",
        GrainLevel.EXTREME,
        "8 mm",
        "8 mm negative — grain is a primary feature of the image, not an artefact.",
    ),
    _FormatRule(
        r"\bsuper\s*16\b",
        GrainLevel.HEAVY,
        "Super 16 mm",
        "Super 16 negative, enlarged to a 2K/4K frame: heavy, coarse grain.",
    ),
    _FormatRule(
        r"\b16\s*mm\b",
        GrainLevel.HEAVY,
        "16 mm",
        "16 mm negative — a quarter of the 35 mm area, so grain is correspondingly large.",
    ),
    _FormatRule(
        r"\btechniscope\b|\b2[\s-]*perf\b|\btwo[\s-]*perf\b",
        GrainLevel.HEAVY,
        "2-perf 35 mm",
        "Techniscope / 2-perf 35 mm uses half the usual frame height, so grain reads "
        "roughly like 16 mm.",
    ),
    _FormatRule(
        r"\bsuper\s*35\b|\b3[\s-]*perf\b",
        GrainLevel.MODERATE,
        "Super 35 mm",
        "Super 35 crops the 35 mm frame for a wide ratio, leaving visible grain.",
    ),
    _FormatRule(
        r"\b(?:65|70)\s*mm\b|\bimax\b|\bvistavision\b|\btodd[\s-]*ao\b|\bultra\s*panavision\b",
        GrainLevel.LIGHT,
        "65 mm / large format",
        "Large-format negative: a big area per frame means fine, tight grain.",
    ),
    _FormatRule(
        r"\b35\s*mm\b|\banamorphic\b|\bpanavision\b|\bcinemascope\b|\btechnicolor\b",
        GrainLevel.MODERATE,
        "35 mm",
        "35 mm negative — moderate grain that a low-bitrate encode will smear.",
    ),
    _FormatRule(
        # "Digital Intermediate" describes the finish, not the capture — a 35 mm film
        # with a DI is still 35 mm, so it must not match here.
        r"\bdigital\b(?!\s+intermediate)|\barri\s*(?:flex\s*)?alexa\b|\balexa\b|\bred\s+(?:one|epic|weapon|komodo|monstro|helium|dragon)\b"
        r"|\bsony\s+(?:venice|cinealta|f6[05]|f55|f35)\b|\bpanavision\s+genesis\b|\bblackmagic\b"
        r"|\bphantom\b|\bdxl\b|\bvenice\b",
        GrainLevel.NONE,
        "digital",
        "Digitally acquired: no photochemical grain, only sensor noise the colourist "
        "may have kept.",
    ),
)

_FILM_CAMERA = re.compile(
    r"\barri\s*flex\b|\bmoviecam\b|\bmitchell\b|\bpanaflex\b|\baaton\b|\bbolex\b|\beclair\b"
    r"|\bimax\s+msm\b|\bvistavision\b",
    re.IGNORECASE,
)
_DIGITAL_INTERMEDIATE = re.compile(r"\bdigital\s+intermediate\b", re.IGNORECASE)

# Rough industry transition: digital capture went from exception to default here.
_DIGITAL_ERA_YEAR = 2012
_FILM_ERA_YEAR = 2000


def _match_formats(entries: tuple[str, ...]) -> list[_FormatRule]:
    """The grainiest rule each entry matches, in table order and without repeats.

    Matching per entry rather than over one joined haystack is what tells
    ``["35 mm (Techniscope)"]`` — one format, named precisely — apart from
    ``["35 mm", "Techniscope"]``, which is a mixed-format production.
    """
    matched: list[_FormatRule] = []
    for entry in entries:
        within = [rule for rule in _FORMAT_RULES if re.search(rule.pattern, entry, re.IGNORECASE)]
        if within:
            grainiest = max(within, key=lambda rule: rule.level.rank)
            if grainiest not in matched:
                matched.append(grainiest)
    return matched


def infer_grain(
    specs: TechnicalSpecs,
    source: SourceMedia,
    *,
    year: int | None = None,
    override: GrainLevel | None = None,
) -> GrainProfile:
    """Best available estimate of the film's grain character.

    Args:
        specs: IMDb technical rows; empty is fine.
        source: The parsed source file, used only to notice that grain may already
            have been destroyed by an earlier encode.
        year: Release year, the fallback when IMDb gave us nothing.
        override: The user's own answer, which always wins.
    """
    if override is not None:
        return GrainProfile(
            level=override,
            confidence=1.0,
            reasons=[f"You set the grain level to {override.value} by hand."],
            user_override=True,
        )

    reasons: list[str] = []
    haystack = specs.all_format_text()
    matches = _match_formats(specs.format_entries())

    level: GrainLevel
    origin: str | None
    confidence: float

    if matches:
        # Several formats can be listed (a mixed-format shoot, or a blow-up). The
        # grainiest one dominates the finished image, so it sets the level; ties go to
        # the more specific rule, which is the earlier one in the table.
        grainiest = max(matches, key=lambda rule: (rule.level.rank, -_FORMAT_RULES.index(rule)))
        level, origin = grainiest.level, grainiest.origin
        reasons.append(grainiest.reason)
        confidence = 0.9 if specs.negative_formats else 0.7

        # Only a difference in *level* makes a production mixed in the way that
        # matters: "35 mm" alongside "Super 35" is one look, "35 mm" alongside
        # "16 mm" is two.
        others = [rule for rule in matches if rule.level is not grainiest.level]
        if others:
            confidence -= 0.1
            listed = ", ".join(dict.fromkeys(rule.origin for rule in others))
            reasons.append(
                f"IMDb also lists {listed}, so this was a mixed-format production — "
                "grain will vary between shots."
            )
    elif _FILM_CAMERA.search(haystack):
        level, origin, confidence = GrainLevel.MODERATE, "35 mm", 0.6
        reasons.append(
            "IMDb lists a film camera but no negative format; assuming 35 mm and moderate grain."
        )
    elif year is not None and year >= _DIGITAL_ERA_YEAR:
        level, origin, confidence = GrainLevel.NONE, "digital", 0.4
        reasons.append(
            f"No IMDb technical data. A {year} release was most likely shot digitally, "
            "so little or no grain is assumed — check the file and correct this if the "
            "picture is grainy."
        )
    elif year is not None and year <= _FILM_ERA_YEAR:
        level, origin, confidence = GrainLevel.MODERATE, "35 mm", 0.4
        reasons.append(
            f"No IMDb technical data. A {year} release was almost certainly 35 mm, so "
            "moderate grain is assumed."
        )
    elif year is not None:
        # Between _FILM_ERA_YEAR and _DIGITAL_ERA_YEAR the year genuinely does not
        # decide it: film and digital capture were both common, so say so.
        level, origin, confidence = GrainLevel.LIGHT, None, 0.25
        reasons.append(
            f"No IMDb technical data, and {year} falls in the years when film and "
            "digital capture were both common, so the year settles nothing. A cautious "
            "light-grain profile is used — set it by hand for a better answer."
        )
    else:
        level, origin, confidence = GrainLevel.LIGHT, None, 0.25
        reasons.append(
            "Nothing to go on — neither IMDb technical data nor a release year — so a "
            "cautious light-grain profile is used. Set it by hand for a better answer."
        )

    if level is not GrainLevel.NONE and _DIGITAL_INTERMEDIATE.search(haystack):
        reasons.append(
            "A digital intermediate is listed. That does not remove grain, but it does "
            "mean the grain you see was signed off in the DI — preserve it rather than "
            "clean it up."
        )

    # A source that has already been squeezed hard has less grain left to protect,
    # and pretending otherwise would spend bitrate on someone else's artefacts.
    video = source.video
    if video is not None and level.rank >= GrainLevel.MODERATE.rank:
        bpp = video.bits_per_pixel
        if bpp is not None and bpp < 0.05:
            confidence -= 0.15
            reasons.append(
                f"The source is only {bpp:.3f} bits per pixel, so much of the grain has "
                "probably already been smoothed away by a previous encode — the grain "
                "settings below protect what is left rather than restore it."
            )

    return GrainProfile(
        level=level,
        confidence=max(0.1, min(confidence, 1.0)),
        reasons=reasons,
        origin_format=origin,
    )
