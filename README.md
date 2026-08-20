# graindamage

Look up a film, confirm which one you meant, describe your source file, and get
encoding settings for **AV1 (SVT-AV1)** and **x265** — HandBrake and FFmpeg — with the
film's own technical characteristics (negative format, aspect ratio, grain) taken into
account rather than guessed at.

Runs as a single Docker image. Python 3.13 · FastAPI · Jinja2 · HTMX.

## Status

Built in milestones. **Milestone 1 (skeleton, Docker, health) is done.**

| # | Milestone                                                  | State   |
| - | ---------------------------------------------------------- | ------- |
| 1 | App skeleton, Docker image, `/healthz`, config reporting   | done    |
| 2 | TMDB provider: title search and pick                       | next    |
| 3 | ffprobe / MediaInfo and IMDb `/technical` parsers          | planned |
| 4 | Deterministic baseline rules (CRF, preset, grain handling) | planned |
| 5 | Gemini structured advice + encoder-flag validation         | planned |
| 6 | HandBrake CLI / FFmpeg command and preset rendering        | planned |
| 7 | Caching, error states, docs polish                         | planned |

## Configuration

Copy `.env.example` to `.env` and fill in what you have. Nothing is mandatory — the
app starts with an empty config and the home page reports which integrations are
active. `/healthz` returns the same information as JSON.

| Variable           | Purpose                                                                             |
| ------------------ | ----------------------------------------------------------------------------------- |
| `TMDB_API_KEY`     | Title search and metadata (themoviedb.org → Settings → API)                         |
| `GEMINI_API_KEY`   | Encoding advice (AI Studio key, tied to your Google Cloud project)                  |
| `GEMINI_MODEL`     | Defaults to `gemini-3.7-flash`                                                      |
| `IMDB_FETCHER_URL` | Optional fetcher for IMDb `/technical` pages; you can paste the page source instead |
| `PORT`             | Defaults to `8080`                                                                  |

Secrets are read from the environment only and never logged. `.env` is gitignored.

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
  main.py               routes: / , /search , /healthz
  config.py             environment-backed settings + capability reporting
  templates/            Jinja2 templates (base, index, partials/)
  static/               css + vendored htmx 2.0.10 (no CDN, works offline)
tests/                  smoke tests for health, index, static, search wiring
.github/workflows/      gitleaks.yml, docker.yml
Dockerfile compose.yaml .env.example
```

## Notes on data sources

- **TMDB** is used for search because IMDb has no public API. Each hit resolves its
  `imdb_id`, so IMDb links and the `/technical` page URL still work.
- **IMDb technical specifications** (aspect ratio, negative format, cinematographic
  process, sound mix) come from the `/technical` page. Since April 2026 IMDb has a WAF in
  front of the site, so the app never scrapes it directly: you paste the page source, or
  point `IMDB_FETCHER_URL` at a fetcher you control.
- **Your source file** matters more than the film's release specs for CRF and bitrate, so
  ffprobe / MediaInfo output can be pasted in and is parsed.

Not affiliated with IMDb. This product uses the TMDB API but is not endorsed or certified
by TMDB.
