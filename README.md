# graindamage

Look up a film, describe your source file, and get encoding settings for **AV1
(SVT-AV1)** and **x265** — HandBrake and FFmpeg — with the film's own technical
characteristics (negative format, aspect ratio, grain) taken into account rather than
guessed at.

Every number comes with the reasoning attached: `CRF 27 = 28 for 1080p, −1 for 35 mm
grain`. Nothing is a black box, and nothing needs an API key — the rules engine has no
configuration and no network, so an empty container still produces real settings.

Runs as a single Docker image. Python 3.13 · FastAPI · Jinja2 · HTMX.

## Status

Built in milestones; all seven are done (v0.7.0).

| # | Milestone                                                        | State |
| - | ---------------------------------------------------------------- | ----- |
| 1 | App skeleton, Docker image, `/healthz`, config reporting         | done  |
| 2 | TMDB provider: title search and pick                             | done  |
| 3 | ffprobe / mkvinfo / MediaInfo and IMDb `/technical` parsers      | done  |
| 4 | Deterministic baseline rules (CRF, preset, grain handling)       | done  |
| 5 | Gemini structured advice + encoder-flag validation               | done  |
| 6 | HandBrake CLI / FFmpeg command and `.json` preset rendering      | done  |
| 7 | Routes, caching, error states, docs polish                       | done  |

## How it works

Three steps in one page, each swapped in over HTMX:

1. **Find the film.** TMDB search (IMDb has no public API); the hit you pick supplies
   its `imdb_id`, which is what finds the negative format.
2. **Describe the source.** Paste `ffprobe`, `mkvinfo` or `mediainfo` output and the
   IMDb `/technical` page source. Every field is overridable by hand.
3. **Get the settings.** Two plans — SVT-AV1 and x265 — each with a CRF derived from a
   resolution anchor and named adjustments, a preset, a validated parameter string, a
   HandBrake command, an FFmpeg command, and a downloadable HandBrake preset.

Grain is inferred from the negative format and process (65 mm and digital-intermediate
sources are clean; Super 16 and pushed 35 mm are not), and it is the single biggest
input to the settings: x265 codes every grain particle, while SVT-AV1 can denoise and
synthesise it back, so the two plans diverge more on a grainy film than a clean one.

### Where the numbers come from

CRF starts at a resolution anchor and moves by signed, labelled adjustments that stay
visible on the plan:

|            | SD | 720p | 1080p | 1440p | 2160p |
| ---------- | -- | ---- | ----- | ----- | ----- |
| SVT-AV1    | 24 | 26   | 28    | 30    | 32    |
| x265       | 19 | 20   | 21    | 22    | 23    |

Then: grain (−3…+1, encoder-dependent), HDR, bit depth, your size and speed
preferences, and the source's own bits-per-pixel — because re-encoding a 0.03 bpp web
rip at a low CRF only reprints someone else's artefacts at a larger size, and the app
says so rather than obliging.

These anchors are conventional community values for 10-bit film encodes, not
measurements. The size estimate is a single exponential fit through one anchor point:
good enough to see that AV1 lands at roughly half the size, not good enough to plan a
disc against.

## Source formats

Three parsers, all optional, all tolerant: a field that cannot be read becomes a
warning, never an error. Paste whichever you have.

| Tool          | Command                                                             | Notes                                                                     |
| ------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **ffprobe**   | `ffprobe -show_format -show_streams -print_format json in.mkv`      | Preferred. Any container, and the only one that reports `pix_fmt`         |
| **mkvinfo**   | `mkvinfo in.mkv`                                                    | Matroska only; authoritative on structure, but bit depth is often absent  |
| **MediaInfo** | `mediainfo in.mkv`                                                  | Any container; human-readable text, locale-dependent number formatting    |

The format is detected from the paste, so there is nothing to select.

### Is mkvinfo a good alternative to ffprobe?

For an MKV, partly — and it is worth knowing exactly where the line is.

**mkvinfo is authoritative for** track types and codec IDs, pixel and display
dimensions, frame rate (from `DefaultDuration`), the entire `Colour` element (primaries,
transfer, matrix, range, chroma siting), MaxCLL/MaxFALL, and ST 2086 mastering
luminance. It reads the container's own metadata rather than a decoder's
interpretation of it, and it picks up per-track bitrate from the `BPS` tags mkvmerge
writes.

**But** `BitsPerChannel` and `ChromaSubsampling` are *optional* Matroska elements, and
most muxers omit them — while bit depth is a primary input to the CRF decision. ffprobe
derives both from `pix_fmt` (`yuv420p10le`), which is always present because the
decoder must know it. And mkvinfo is MKV-only: no MP4, no MOV, no TS.

So: ffprobe first, mkvinfo when it is what you have or when you want the container's
own colour signalling, and every parser reports what it could not determine so you can
fill the gap by hand.

## IMDb technical specifications

Aspect ratio, negative format, cinematographic process and printed film format come
from the film's `/technical` page. Since April 2026 IMDb has a WAF in front of the
site, so **the app never scrapes it**. Two ways to get the data in:

- **Paste the page source.** Open `https://www.imdb.com/title/tt0083658/technical`,
  view source, paste. Both the current `__NEXT_DATA__` payload and the older markup are
  parsed. This always works and needs no configuration.
- **Point `IMDB_FETCHER_URL` at a fetcher you run.** The app sends
  `GET <IMDB_FETCHER_URL>?url=https://www.imdb.com/title/<tt…>/technical/`, adding
  `Authorization: Bearer <IMDB_FETCHER_TOKEN>` when a token is set, and expects the page
  source as the response body. A JSON response is unwrapped from any of `content`,
  `html`, `body`, `data`, `result` or `text`, so browserless and most scraping APIs work
  unmodified. A proxy, a Worker, or anything cookie-bearing you control will do.

A pasted page always wins over a fetched one.

If a fetch fails the app says so and carries on with the release year as the only clue
to the grain, which it labels as a guess.

## Gemini (optional)

With `GEMINI_API_KEY` set, a checkbox appears that asks Gemini to review the
deterministic plan. It can adjust CRF, preset, tune and parameters, and it writes the
prose. It is never trusted:

- Everything it returns passes an **allowlist** in [`app/advice/validate.py`](app/advice/validate.py):
  a parameter is dropped unless it is a known parameter for that encoder *and* its value
  is in range *and* the value contains no shell-interesting characters.
- CRF may move at most **4 points** from the baseline, and stays inside the encoder's
  usable range.
- Parameters derived from your file's own colour signalling are **restored after
  validation** — the model cannot see the file, so it has no business changing them.
- Everything dropped is shown to you, so a disagreement is visible rather than silent.

The prompt deliberately excludes your local input path. Commands are assembled with
`shlex.join`, never string interpolation.

Without a key, the rules engine answers alone and the page says so.

## Configuration

Copy `.env.example` to `.env` and fill in what you have. Nothing is mandatory: the app
starts with an empty config, the home page reports which integrations are active, and
`/healthz` returns the same as JSON.

| Variable                       | Default                          | Purpose                                                        |
| ------------------------------ | -------------------------------- | -------------------------------------------------------------- |
| `TMDB_API_KEY`                 | —                                | Title search and metadata (themoviedb.org → Settings → API)    |
| `TMDB_LANGUAGE`                | `en-US`                          | Metadata language                                              |
| `TMDB_BASE_URL`                | `https://api.themoviedb.org/3`   | API root                                                       |
| `TMDB_IMAGE_BASE_URL`          | `https://image.tmdb.org/t/p`     | Poster host                                                    |
| `TMDB_POSTER_SIZE`             | `w185`                           | Poster size segment                                            |
| `TMDB_TIMEOUT_SECONDS`         | `10.0`                           | Per-request timeout                                            |
| `TMDB_MAX_RESULTS`             | `8`                              | Search hits offered (1–20)                                     |
| `GEMINI_API_KEY`               | —                                | Encoding advice (AI Studio key)                                |
| `GEMINI_MODEL`                 | `gemini-3.7-flash`               | Model name                                                     |
| `GEMINI_BASE_URL`              | `…/v1beta`                       | Generative Language API root                                   |
| `GEMINI_TIMEOUT_SECONDS`       | `45.0`                           | Per-request timeout                                            |
| `GEMINI_MAX_OUTPUT_TOKENS`     | `2048`                           | Response cap                                                   |
| `GEMINI_TEMPERATURE`           | `0.2`                            | Low on purpose: this is a settings decision                    |
| `IMDB_FETCHER_URL`             | —                                | Optional `/technical` fetcher; pasting works without one       |
| `IMDB_FETCHER_TOKEN`           | —                                | Sent as `Authorization: Bearer …` when set                     |
| `IMDB_FETCHER_TIMEOUT_SECONDS` | `20.0`                           | Per-request timeout                                            |
| `CACHE_TTL_SECONDS`            | `3600`                           | In-process TTL for TMDB, IMDb and Gemini results; `0` disables |
| `HOST` / `PORT`                | `0.0.0.0` / `8080`               | Listen address                                                 |
| `USER_AGENT`                   | `graindamage/0.7 …`              | Sent on every outbound request                                 |
| `DEBUG`                        | `false`                          | Verbose errors                                                 |

Secrets are read from the environment only and never logged. The Gemini key travels in
an `x-goog-api-key` header rather than a query string, so it cannot land in a proxy
log. `.env` is gitignored, and CI scans the history for leaked secrets.

## Routes

| Route        | Method | Returns                                                          |
| ------------ | ------ | ---------------------------------------------------------------- |
| `/`          | GET    | The page                                                         |
| `/healthz`   | GET    | `{"status": "ok", "version": …, "capabilities": {…}}`            |
| `/search`    | POST   | Search results partial                                           |
| `/pick`      | POST   | The film, its IMDb specs, and the source form                    |
| `/technical` | POST   | Fetched IMDb specs partial (fetcher only)                        |
| `/advise`    | POST   | Both plans, with commands                                        |
| `/preset`    | POST   | HandBrake `.json` preset as a download                           |
| `/api/docs`  | GET    | OpenAPI docs                                                     |

Every error path answers **HTTP 200** with an explanation partial, because HTMX does
not swap a non-2xx response into the page. A dead TMDB, a refused fetcher, an
unparseable paste and a silent Gemini each cost a warning, never the settings.

## Run with Docker

```bash
docker compose up --build      # http://localhost:8080
```

Or without compose:

```bash
docker build -t graindamage .
docker run --rm -p 8080:8080 --env-file .env graindamage
```

The image runs as a non-root user and ships a `HEALTHCHECK` against `/healthz`.

## Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8080
```

`uv` works too, if you have it (`uv sync --extra dev`, or `uv run uvicorn app.main:app --reload`).
The Docker build uses `uv` regardless — it pulls the binary from the official image, so no
local install is required.

Checks:

```bash
ruff check . && ruff format --check .
mypy app tests
pytest
```

The tests reach no network: providers take an injected `httpx2.AsyncClient`, so
`tests/support.py` hands them a `MockTransport`, and settings are built with
`_env_file=None` so a local `.env` cannot switch an integration on mid-test.

## CI

Two workflows, both in [`.github/workflows/`](.github/workflows):

**`gitleaks.yml`** — secret scanning on every push to `main`, every pull request, weekly,
and on demand. It runs the pinned upstream CLI image (`ghcr.io/gitleaks/gitleaks:v8.30.1`)
twice: once over the **full git history** (`gitleaks git`, checkout with `fetch-depth: 0`)
and once over the **working tree** (`gitleaks dir`). Findings are redacted, fail the job,
and are uploaded as a SARIF artifact (`gitleaks-reports`, 30 days).

The official `gitleaks-action` is deliberately not used: it requires a licence key for
organisation-owned repositories and is no longer MIT-licensed. The CLI image has neither
constraint and pins an exact version.

Run the same scans locally:

```bash
docker run --rm -v "$PWD:/repo" ghcr.io/gitleaks/gitleaks:v8.30.1 git /repo --redact --verbose
docker run --rm -v "$PWD:/repo" ghcr.io/gitleaks/gitleaks:v8.30.1 dir /repo --redact --verbose
```

**`docker.yml`** — builds the image and pushes it to Docker Hub as
`<DOCKERHUB_USERNAME>/graindamage`:

| Trigger         | Behaviour                                            |
| --------------- | ---------------------------------------------------- |
| push to `main`  | build → smoke test → push `<short-sha>` and `latest` |
| pull request    | build → smoke test only, no login and no push        |
| manual dispatch | same as push                                         |

Tags come from `docker/metadata-action`: the 7-character commit SHA as the build tag, plus
`latest` on the default branch only. Pushed images are multi-arch (`linux/amd64`,
`linux/arm64`). Before anything is pushed, an amd64 image is loaded into the runner and
started, and `/healthz` must answer `{"status": "ok"}` — so a broken build cannot take over
the `latest` tag.

Required repository secrets:

| Secret               | Purpose                                                          |
| -------------------- | ---------------------------------------------------------------- |
| `DOCKERHUB_USERNAME` | Docker Hub account, also the image namespace                     |
| `DOCKERHUB_TOKEN`    | Docker Hub access token (Read & Write), not the account password |

Because the username is a secret, it appears masked as `***` in logs and job summaries. If
you would rather read the real tags there, move it to a repository *variable* and swap
`secrets.DOCKERHUB_USERNAME` for `vars.DOCKERHUB_USERNAME` in the workflow.

## Layout

```text
app/
  main.py               application factory: settings in, wired app out
  routes.py             every HTTP handler, form parsing, error partials
  config.py             environment-backed settings + capability reporting
  models.py             the domain: tracks, specs, plans, advice
  cache.py              async TTL cache, one per provider
  advice/
    grain.py            negative format and process → grain profile
    rules.py            the deterministic baseline: CRF, preset, parameters
    validate.py         allowlist for anything a model returns
    encoders.py         HandBrake and FFmpeg commands, HandBrake presets
  providers/
    tmdb.py             title search and metadata
    imdb.py             /technical parsing, paste or fetcher
    gemini.py           structured advice, merged over the baseline
  sources/
    ffprobe.py mkvinfo.py mediainfo.py    the three source parsers
    parsing.py          tolerant number and unit parsing shared by them
    colors.py           colour primaries / transfer / matrix normalisation
  templates/            Jinja2 templates (base, index, partials/)
  static/               css, a little js, vendored htmx 2.0.10 (no CDN)
tests/                  one module per app module, plus fixtures/
.github/workflows/      gitleaks.yml, docker.yml
Dockerfile compose.yaml .env.example
```

## Notes on data sources

- **TMDB** is used for search because IMDb has no public API. Each hit resolves its
  `imdb_id`, so IMDb links and the `/technical` page URL still work.
- **IMDb technical specifications** are never scraped: paste the page source, or point
  `IMDB_FETCHER_URL` at a fetcher you control.
- **Your source file** matters more than the film's release specs for CRF and bitrate,
  which is why all three source formats are parsed and every field can be overridden.

Not affiliated with IMDb. This product uses the TMDB API but is not endorsed or certified
by TMDB.
