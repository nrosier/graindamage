"""The shared vocabulary: what a film is, what a source file is, what we advise.

Everything crossing a module boundary is one of these models. Providers build them,
the rules engine consumes them, the templates render them, and the Gemini schema is
derived from them — so a field added here shows up everywhere at once.

Colour and HDR values are normalised to **FFmpeg's spelling** (``bt2020nc``,
``smpte2084``, ``tv``) regardless of which tool produced the input, because that is
also what x265 wants. SVT-AV1 wants H.273 integers instead; :mod:`app.sources.colors`
holds the mapping in that direction.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator

# --- films ------------------------------------------------------------------


class MovieHit(BaseModel):
    """One search result, cheap enough to render a whole list of."""

    tmdb_id: int
    title: str
    original_title: str | None = None
    year: int | None = None
    overview: str | None = None
    poster_url: str | None = None
    vote_average: float | None = None

    @property
    def display_title(self) -> str:
        if self.original_title and self.original_title != self.title:
            return f"{self.title} ({self.original_title})"
        return self.title


class Movie(MovieHit):
    """A confirmed pick, with the extra detail a second request buys us."""

    imdb_id: str | None = None
    runtime_minutes: int | None = None
    release_date: str | None = None
    genres: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    original_language: str | None = None

    @property
    def imdb_url(self) -> str | None:
        return f"https://www.imdb.com/title/{self.imdb_id}/" if self.imdb_id else None

    @property
    def imdb_technical_url(self) -> str | None:
        return f"https://www.imdb.com/title/{self.imdb_id}/technical/" if self.imdb_id else None


class TechnicalSpecs(BaseModel):
    """IMDb's ``/technical`` page, reduced to the rows that affect an encode.

    Every row is a list because IMDb lists one entry per release or per camera
    (``35 mm`` *and* ``Digital`` for a mixed-format shoot, for instance), and the
    grain heuristic wants to see all of them.
    """

    aspect_ratios: list[str] = Field(default_factory=list)
    negative_formats: list[str] = Field(default_factory=list)
    cinematographic_processes: list[str] = Field(default_factory=list)
    printed_formats: list[str] = Field(default_factory=list)
    cameras: list[str] = Field(default_factory=list)
    laboratories: list[str] = Field(default_factory=list)
    film_lengths: list[str] = Field(default_factory=list)
    sound_mixes: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    runtimes: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(getattr(self, name) for name in self.__class__.model_fields)

    def format_entries(self) -> tuple[str, ...]:
        """The rows the grain heuristic reads, one lowercased entry at a time.

        Kept separate rather than pre-joined because one entry naming two formats
        (``35 mm (Techniscope)``) is a single format described precisely, while two
        entries naming two formats is a mixed-format shoot — and those want
        different answers.
        """
        parts = (
            *self.negative_formats,
            *self.cinematographic_processes,
            *self.printed_formats,
            *self.cameras,
            *self.film_lengths,
        )
        return tuple(entry.lower() for entry in parts if entry.strip())

    def all_format_text(self) -> str:
        """Everything the grain heuristic reads, lowercased into one haystack."""
        return " | ".join(self.format_entries())


# --- source files -----------------------------------------------------------


class SourceTool(StrEnum):
    """Which utility produced the text the user pasted."""

    FFPROBE = "ffprobe"
    MKVINFO = "mkvinfo"
    MEDIAINFO = "mediainfo"


class MasteringDisplay(BaseModel):
    """SMPTE ST 2086 mastering display metadata, in real units.

    Kept as CIE xy floats and cd/m² rather than either encoder's wire format,
    because x265 wants 0.00002-unit integers and SVT-AV1 wants these floats.
    """

    red_x: float
    red_y: float
    green_x: float
    green_y: float
    blue_x: float
    blue_y: float
    white_x: float
    white_y: float
    max_luminance: float
    min_luminance: float

    def to_x265(self) -> str:
        """``G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)`` in 0.00002 / 0.0001 units."""

        def xy(value: float) -> int:
            return round(value * 50_000)

        def lum(value: float) -> int:
            return round(value * 10_000)

        return (
            f"G({xy(self.green_x)},{xy(self.green_y)})"
            f"B({xy(self.blue_x)},{xy(self.blue_y)})"
            f"R({xy(self.red_x)},{xy(self.red_y)})"
            f"WP({xy(self.white_x)},{xy(self.white_y)})"
            f"L({lum(self.max_luminance)},{lum(self.min_luminance)})"
        )

    def to_svt_av1(self) -> str:
        """The same string, but with the actual float values SVT-AV1 expects."""

        def num(value: float) -> str:
            return f"{value:g}"

        return (
            f"G({num(self.green_x)},{num(self.green_y)})"
            f"B({num(self.blue_x)},{num(self.blue_y)})"
            f"R({num(self.red_x)},{num(self.red_y)})"
            f"WP({num(self.white_x)},{num(self.white_y)})"
            f"L({num(self.max_luminance)},{num(self.min_luminance)})"
        )


HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


class VideoTrack(BaseModel):
    index: int | None = None
    codec: str | None = None  # normalised: hevc, h264, av1, mpeg2video, vc1, prores…
    profile: str | None = None
    width: int | None = None
    height: int | None = None
    display_aspect_ratio: str | None = None
    frame_rate: float | None = None
    frame_rate_mode: str | None = None  # constant / variable
    bit_depth: int | None = None
    chroma_subsampling: str | None = None  # 4:2:0 / 4:2:2 / 4:4:4
    pix_fmt: str | None = None
    scan_type: str | None = None  # progressive / interlaced
    bitrate_bps: int | None = None
    color_range: str | None = None  # tv / pc
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_matrix: str | None = None
    mastering_display: MasteringDisplay | None = None
    max_cll: int | None = None
    max_fall: int | None = None
    dolby_vision: bool = False
    hdr10_plus: bool = False

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS

    @property
    def is_wide_gamut(self) -> bool:
        return self.color_primaries in {"bt2020", "smpte431", "smpte432"}

    @property
    def is_interlaced(self) -> bool:
        return bool(self.scan_type and self.scan_type.lower().startswith("inter"))

    @property
    def pixels(self) -> int | None:
        if self.width and self.height:
            return self.width * self.height
        return None

    @property
    def bits_per_pixel(self) -> float | None:
        """Source bitrate per pixel per frame — our proxy for "how good is this?".

        Roughly: >0.20 is a disc-quality HEVC/AVC master, 0.10–0.20 a good web
        release, <0.05 something already squeezed hard enough that re-encoding
        mostly reprints its artefacts.
        """
        pixels = self.pixels
        if not (self.bitrate_bps and pixels and self.frame_rate):
            return None
        return self.bitrate_bps / (pixels * self.frame_rate)


class AudioTrack(BaseModel):
    index: int | None = None
    codec: str | None = None
    channels: int | None = None
    channel_layout: str | None = None
    sample_rate: int | None = None
    bitrate_bps: int | None = None
    language: str | None = None
    title: str | None = None
    default: bool = False

    @property
    def is_lossless(self) -> bool:
        return (self.codec or "").lower() in {
            "truehd",
            "dts-hd ma",
            "dtshd_ma",
            "flac",
            "pcm",
            "alac",
            "mlp",
        }


class SubtitleTrack(BaseModel):
    index: int | None = None
    codec: str | None = None
    language: str | None = None
    title: str | None = None
    forced: bool = False


class SourceMedia(BaseModel):
    container: str | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None
    overall_bitrate_bps: int | None = None
    video: VideoTrack | None = None
    audio: list[AudioTrack] = Field(default_factory=list)
    subtitles: list[SubtitleTrack] = Field(default_factory=list)

    @property
    def resolution_label(self) -> str | None:
        if not self.video or not self.video.height:
            return None
        height = self.video.height
        if height <= 576:
            return "SD"
        if height <= 720:
            return "720p"
        if height <= 1080:
            return "1080p"
        if height <= 1440:
            return "1440p"
        return "2160p"


class SourceReport(BaseModel):
    """A parse result plus an honest account of what could not be determined."""

    tool: SourceTool
    media: SourceMedia
    warnings: list[str] = Field(default_factory=list)

    @property
    def has_video(self) -> bool:
        return self.media.video is not None


# --- grain ------------------------------------------------------------------


class GrainLevel(StrEnum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    EXTREME = "extreme"

    @property
    def rank(self) -> int:
        return _GRAIN_RANKS[self]

    @classmethod
    def from_rank(cls, rank: int) -> GrainLevel:
        clamped = max(0, min(rank, len(_GRAIN_ORDER) - 1))
        return _GRAIN_ORDER[clamped]


_GRAIN_ORDER: tuple[GrainLevel, ...] = (
    GrainLevel.NONE,
    GrainLevel.LIGHT,
    GrainLevel.MODERATE,
    GrainLevel.HEAVY,
    GrainLevel.EXTREME,
)
_GRAIN_RANKS: dict[GrainLevel, int] = {level: i for i, level in enumerate(_GRAIN_ORDER)}


class GrainProfile(BaseModel):
    level: GrainLevel
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    origin_format: str | None = None  # "35 mm", "16 mm", "digital"…
    user_override: bool = False


# --- advice -----------------------------------------------------------------


class Encoder(StrEnum):
    SVT_AV1 = "svt-av1"
    X265 = "x265"

    @property
    def label(self) -> str:
        return "AV1 (SVT-AV1)" if self is Encoder.SVT_AV1 else "x265 (HEVC)"


class SpeedPreference(StrEnum):
    """How much CPU time the user is willing to spend."""

    QUALITY = "quality"
    BALANCED = "balanced"
    FAST = "fast"


class SizePreference(StrEnum):
    """Where to sit on the size/fidelity curve."""

    ARCHIVAL = "archival"
    BALANCED = "balanced"
    COMPACT = "compact"


class SpecsSource(StrEnum):
    """Where a set of technical rows came from, because it changes how much they weigh.

    Pasted rows are IMDb's own words. Looked-up rows are a language model's recollection
    of that page, which is worth having and worth labelling as such.
    """

    PASTED = "pasted"
    GEMINI = "gemini"


class EncodeRequest(BaseModel):
    """Everything the rules engine needs, assembled from the step-2 form."""

    movie: Movie | None = None
    specs: TechnicalSpecs = Field(default_factory=TechnicalSpecs)
    specs_source: SpecsSource | None = None
    source: SourceMedia = Field(default_factory=SourceMedia)
    source_tool: SourceTool | None = None
    grain_override: GrainLevel | None = None
    bit_depth_override: int | None = None
    speed: SpeedPreference = SpeedPreference.BALANCED
    size: SizePreference = SizePreference.BALANCED
    input_path: str = "input.mkv"
    output_stem: str = "output"


class Adjustment(BaseModel):
    """One traceable step in arriving at a CRF, so the number is arguable."""

    label: str
    delta: float
    detail: str | None = None


class EncoderPlan(BaseModel):
    encoder: Encoder
    crf: float
    preset: str
    # x265 only: its ``--tune`` is a named bundle set outside the parameter string,
    # whereas SVT-AV1's tune is a number and lives in ``params``.
    tune: str | None = None
    params: dict[str, str] = Field(default_factory=dict)
    pixel_format: str = "yuv420p10le"
    adjustments: list[Adjustment] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    ffmpeg_command: str = ""
    handbrake_command: str = ""
    estimated_bitrate_bps: int | None = None

    @property
    def params_string(self) -> str:
        """``key=value:key=value`` — the form both ``-svtav1-params`` and
        ``-x265-params`` take."""
        return ":".join(f"{k}={v}" for k, v in self.params.items())

    @property
    def crf_label(self) -> str:
        """``27`` rather than ``27.0``, but ``26.5`` when it really is a half step."""
        return str(int(self.crf)) if float(self.crf).is_integer() else f"{self.crf:g}"

    @property
    def estimated_size_note(self) -> str | None:
        if not self.estimated_bitrate_bps:
            return None
        return f"~{self.estimated_bitrate_bps / 1_000_000:.1f} Mb/s video"


class AdviceSource(StrEnum):
    BASELINE = "baseline"
    GEMINI = "gemini"


class Advice(BaseModel):
    """The finished answer: two plans, plus how we got here."""

    source: AdviceSource = AdviceSource.BASELINE
    grain: GrainProfile
    plans: list[EncoderPlan] = Field(default_factory=list)
    summary: str | None = None
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rejected_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _at_least_one_plan(self) -> Self:
        if not self.plans:
            raise ValueError("advice needs at least one encoder plan")
        return self

    def plan_for(self, encoder: Encoder) -> EncoderPlan | None:
        return next((p for p in self.plans if p.encoder is encoder), None)
