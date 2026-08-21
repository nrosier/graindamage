"""``python -m app.cli`` — the same entry point as the installed ``graindamage``."""

from __future__ import annotations

import sys

from app.cli.app import main

sys.exit(main())
