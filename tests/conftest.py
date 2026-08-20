"""Fixtures every module can rely on."""

from __future__ import annotations

import pytest

from app.models import SourceReport
from app.sources import parse_source
from tests.support import fixture


@pytest.fixture
def ffprobe_report() -> SourceReport:
    """The 2160p HDR ffprobe fixture, parsed."""
    return parse_source(fixture("ffprobe_uhd_hdr.json"))


@pytest.fixture
def mkvinfo_report() -> SourceReport:
    return parse_source(fixture("mkvinfo_uhd_hdr.txt"))


@pytest.fixture
def mediainfo_report() -> SourceReport:
    return parse_source(fixture("mediainfo_1080p_film.txt"))
