# graindamage

Look up a film, describe your source file, and get encoding settings for **AV1
(SVT-AV1)** and **x265** — HandBrake and FFmpeg — with the film's own technical
characteristics (negative format, aspect ratio, grain) taken into account rather than
guessed at.

Every number comes with the reasoning attached: `CRF 27 = 28 for 1080p, −1 for 35 mm
grain`. Nothing is a black box, and nothing needs an API key — the rules engine has no
configuration and no network, so an empty container still produces real settings.

Two front-ends over one engine: a web page, and a command line that takes a file and
writes a HandBrake preset and an FFmpeg script beside it. Both ship in a single Docker
image. Python 3.13 · FastAPI · Jinja2 · HTMX.

## Status

Built in milestones; all eight are done (v0.8.0).

| # | Milestone                                                        | State |
| - | ---------------------------------------------------------------- | ----- |
| 1 | App skeleton, Docker image, `/healthz`, config reporting         | done  |
| 2 | TMDB provider: title search and pick                             | done  |
| 3 | ffprobe / mkvinfo / MediaInfo and IMDb `/technical` parsers      | done  |
| 4 | Deterministic baseline rules (CRF, preset, grain handling)       | done  |
| 5 | Gemini structured advice + encoder-flag validation               | done  |
| 6 | HandBrake CLI / FFmpeg command and `.json` preset rendering      | done  |
| 7 | Routes, caching, error states, docs polish                       | done  |
| 8 | Command line: filename → ffprobe → film picker → two files       | done  |

## How it works

Three steps in one page, each swapped in over HTMX:

1. **Find the film.** TMDB search (IMDb has no public API); the hit you pick supplies
   its `imdb_id`, which is what finds the negative format.
2. **Describe the source.** Paste `ffprobe`, `mkvinfo` or `mediainfo` output. The
   IMDb technical rows arrive from a Gemini look-up, or you paste those too. Every
   field is overridable by hand.
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

### Grain tuning

Grain is what the settings are *for*, so both plans act on it directly rather than
leaving it to the CRF. On SVT-AV1 the strength is the easy part and the denoise flag is
the real decision:

| Grain    | `film-grain` | `film-grain-denoise` | Effect                                       |
| -------- | ------------ | -------------------- | -------------------------------------------- |
| light    | 4            | 0                    | real grain coded, synthesis as a floor       |
| moderate | 8            | 0                    | real grain coded, synthesis as a floor       |
| heavy    | 12           | 1                    | denoise and re-synthesise — grain *is* the image |
| extreme  | 20           | 1                    | denoise and re-synthesise                    |

Below heavy grain, synthesis is a **floor over the coded grain, not a replacement for
it**: with `film-grain-denoise=0` the grain in the picture is still encoded, and the
synthesised layer only fills in where the quantiser flattened it — so a 35 mm look is
never traded for a uniform synthetic field. `enable-restoration=0` goes with that path,
because AV1's loop restoration is a Wiener filter, i.e. a denoiser, and it would smooth
away what was just paid for. Where synthesis is on at all, the preset is capped at 6,
which is where SVT-AV1 itself starts warning against film grain.

x265 codes every particle instead. From moderate up it runs `--tune grain`, which raises
psy-rd to 4.0 and psy-rdoq to 10 and turns off SAO, cu-tree, AQ and rskip — but, despite
its reputation, does *not* touch qcomp or the deblocking offsets, so those are set by
hand (`qcomp=0.8`, `deblock=-1`), along with `strong-intra-smoothing=0` on a film source.

None of it applies to a clean source: synthesising grain over an image that never had any
just lays noise on it, so a digital origin format gets no grain parameters at all.

**Older negatives are tuned harder.** Stocks got finer, and dupe negatives and optical
printing got rarer, so the same `35 mm` row means coarser, higher-contrast grain on a
1962 film than on a 1998 one. At or before 1970, a photochemical source gets `film-grain`
+4 and `qp-scale-compress-strength=2` on AV1, and `qcomp=0.85` with `deblock=-2` on x265,
each with the year named in the rationale. The era moves the tuning only — the inferred
grain level, the CRF, the preset and the size estimate are all unchanged.

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
from the film's `/technical` page, and they are what decide the grain settings. Since
April 2026 IMDb has an AWS WAF in front of the site that answers anything automated
with a JavaScript challenge, so **the app never requests imdb.com**. The rows arrive
one of two other ways.

### Ask Gemini for them

Pick a film and the app asks Gemini, in words, for the page:

> Fetch the technical specifications of *Blade Runner* (1982), IMDb id tt0083658, from
> IMDb: `https://www.imdb.com/title/tt0083658/technical/`

The answer comes back as the same ten rows the page has, and appears under the film
immediately. It is cached, so a reload or a tweaked preference costs nothing. The
button needs `GEMINI_API_KEY`; there is nothing else to configure and nothing to run.

**It is a model answering, not IMDb.** So the app never pretends otherwise:

- The look-up reports its own `confidence` (`high`, `medium`, `low`) and the page
  prints it, along with whether the rows were searched for or recalled.
- The prompt tells the model that an empty row is the right answer for a row it does
  not know, and forbids inventing a camera, laboratory or process to fill one out. An
  honest blank asks you for a paste; a plausible wrong negative format would quietly
  misdirect the grain settings, which is the worse failure.
- The advice page repeats where the rows came from, because grain is the one decision
  they drive.

`GEMINI_WEB_GROUNDING` (on by default) lets the model search the web for the page
rather than answer from training data. Google's API does not allow search and a
response schema on the same request, so a grounded look-up asks for JSON in words and
is parsed leniently; if the model or the key refuses the tool (HTTP 400), the app
retries once without it, schema-constrained. Turning grounding off makes that the only
path.

### Paste them

Beside the film is a direct link to its `/technical` page and a box to paste into.
Open the page, select the specifications, copy, paste — the formatted text as it reads
on screen parses, tabs, `·` separators, `Runtime: 1 hour 57 minutes` on one line and
all. So does the page source, whether it carries the current `__NEXT_DATA__` payload,
the current server-rendered markup, or the pre-2020 table. Page furniture (`Edit`,
`More to explore`, `Recently viewed`) and prose are discarded.

**A paste always wins over a look-up**, and no caveat is attached to one — those rows
are IMDb's own words.

With neither, the release year is the only clue to the grain, and the app labels it as
a guess.

## Gemini (optional)

With `GEMINI_API_KEY` set, a checkbox appears that asks Gemini to review the
deterministic plan. It can adjust CRF, preset, tune and parameters, and it writes the
prose. On a film source the prompt requires it to answer with grain parameters and
era-specific reasoning — which stock, which generation of print, `film-grain-denoise` one
way or the other and why — and says outright that a plan moving the CRF and nothing else
is not an answer. It is never trusted:

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
| `GEMINI_WEB_GROUNDING`         | `true`                           | Let the specs look-up search the web instead of recalling      |
| `CACHE_TTL_SECONDS`            | `3600`                           | In-process TTL for TMDB and Gemini results; `0` disables       |
| `HOST` / `PORT`                | `0.0.0.0` / `8080`               | Listen address                                                 |
| `USER_AGENT`                   | `graindamage/0.8 …`              | Sent on every outbound request                                 |
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
| `/technical` | POST   | The IMDb specs partial, from a Gemini look-up                    |
| `/advise`    | POST   | Both plans, with commands                                        |
| `/preset`    | POST   | HandBrake `.json` preset as a download                           |
| `/api/docs`  | GET    | OpenAPI docs                                                     |

Every error path answers **HTTP 200** with an explanation partial, because HTMX does
not swap a non-2xx response into the page. A dead TMDB, a look-up that knows
nothing, an unparseable paste and a silent Gemini each cost a warning, never the
settings.

## Command line

One command, one argument, and it asks you the rest:

```bash
graindamage /media/Blade.Runner.1982.2160p.UHD.BluRay.DV.HDR.x265-GRP.mkv
```

That is the whole interface. It reads the title out of the filename, runs `ffprobe` on the
file itself, and then walks you through four steps — each one a list you move a cursor
through. `python -m app.cli <file>` is the same thing without installing the script.

### Inside the container

The image already carries `ffprobe`, so this is the shortest way in. `compose.yaml` mounts
`MEDIA_DIR` (default `./media`) at `/media`, so there is something to point at:

```bash
echo 'MEDIA_DIR=/srv/films' >> .env    # your library
docker compose up -d --build

docker compose exec -u "$(id -u):$(id -g)" graindamage /bin/sh
$ graindamage /media/Blade.Runner.1982.2160p.mkv
```

`graindamage` is on the `PATH` in there; `-u` is what makes the two written files belong to
you rather than to the image's `app` user. One line, without the shell:

```bash
docker compose exec -u "$(id -u):$(id -g)" graindamage graindamage /media/film.mkv
```

Or with plain Docker, mounting the directory you are standing in:

```bash
docker run --rm -it --env-file .env \
  -v "$PWD:/media" --user "$(id -u):$(id -g)" \
  graindamage graindamage /media/film.mkv
```

`-it` is what gives the menus a terminal. Without one — a pipe, `docker exec` with `-T`,
`docker run` without `-it` — every list falls back to a numbered prompt read one line at a
time, and `--yes` skips the asking altogether.

### The four steps

Arrows or `k`/`j` move, `1`–`9` jump, Enter picks, `q` / Esc / Ctrl-C stops. The first row
is always the way forward, so with both keys set **Enter four times is a complete run**: the
film the filename agrees with, the rows looked up, the settings as inferred, written. A
missing key takes rows away rather than adding work — there is never a row that cannot do
anything.

```text
graindamage 0.8.0 — 4 steps to two files.

File     Blade.Runner.1982.2160p.UHD.BluRay.DV.HDR.x265-GRP.mkv
Source   3840×2160 · hevc · 10-bit · HDR · 23.976 fps · 62.0 Mb/s · 1h 56m · 26.3 GB · 1 audio
Name     Blade Runner (1982) — read from the filename

Step 1 of 4 · Which film is this?
   Blade Runner 2049  2017
 ▸ Blade Runner  1982  · matches the filename
   Search for another title  the filename is only a guess
   Type the film's IMDb id  tt… — enough on its own for the rows
   No film — settings from the file alone  grain then comes from the release year
   Stop, and write nothing
  ↑↓ move · 1-9 jump · enter choose · q abort
```

The cursor starts on the hit the filename's year agrees with, not on TMDB's first answer —
which for this file is the 2049 sequel. An IMDb id typed here (the address works too, the
id is read out of it) is enough on its own: it gets the technical rows with no TMDB key at
all.

**Step 2** is where the technical rows come from, and every route to them is on it:

```text
The aspect ratio, negative format and process are what decide the grain settings.
  https://www.imdb.com/title/tt0083658/technical/

Step 2 of 4 · The film's technical specifications
 ▸ Ask Gemini for the rows  gemini-3.7-flash — a look-up, not IMDb
   Paste the technical page yourself  https://www.imdb.com/title/tt0083658/technical/
   Read the page out of a file
   Skip — guess the grain from the release year  1982
   Stop, and write nothing
  ↑↓ move · 1-9 jump · enter choose · q abort
```

A look-up prints what it got and offers to keep it, so a wrong row can be replaced by a
paste; the paste box prints the `/technical` address first and ends on Ctrl-D (Ctrl-D on
its own skips). Paste beats look-up, always. Whichever route you took, the report says
which one it was. See [IMDb technical specifications](#imdb-technical-specifications) for
why IMDb is never fetched directly.

**Step 3** is the encode, as a hub: pick a line, change it, come back to the hub.

```text
Step 3 of 4 · The encode
 ▸ These are fine  on to the last step
   Speed           balanced — SVT-AV1 preset 4, x265 slow
   Size            balanced — the CRF the resolution anchors at
   Grain           moderate (Super 35 mm) — inferred, confidence 90%
   Bit depth       from the source — 10-bit
   Encoder order   AV1 (SVT-AV1) first
   Gemini review   on — it may move the CRF, and writes the prose
   Stop, and write nothing
  ↑↓ move · 1-9 jump · enter choose · q abort
```

Each sub-menu opens with the cursor on the value in force now, and the grain line is
recomputed as you go — so setting the film's format by hand and watching the level move is
the same one keypress as leaving it alone.

**Step 4** shows what is about to happen and lets you go back to any of the three:

```text
Ready.
  Film     Blade Runner (1982) · tt0083658
  Rows     Gemini web search · confidence medium
  Grain    moderate (Super 35 mm) — inferred, confidence 90%
  Encode   balanced size, balanced speed, AV1 (SVT-AV1) first, Gemini review on
  Write    /media/Blade.Runner.1982.2160p.UHD.BluRay.DV.HDR.x265-GRP.graindamage.json
  Write    /media/Blade.Runner.1982.2160p.UHD.BluRay.DV.HDR.x265-GRP.graindamage.sh

Step 4 of 4 · Write the two files?
 ▸ Write them  a HandBrake preset and an FFmpeg script
   Back to the film
   Back to the technical rows
   Back to the encode settings
   Stop, and write nothing
```

Nothing is written until that Enter, nothing is overwritten without being asked — files
already in the way are the *first* question, before any network call — and prompts go to
stderr while the report goes to stdout, so `graindamage film.mkv > notes.txt` still shows
you the menus.

### No argument at all

`graindamage` on its own offers what it can see: the current directory first, then
`/media`, `/movies`, `/films`, `/video`, `/videos`, `/data` and `/mnt`, breadth-first, three
directories down, twenty files at most.

```text
Which file? These are the ones I can see
 ▸ films/Alien.1979.2160p.UHD.BluRay.mkv  41.5 GB
   films/Blade.Runner.1982.2160p.UHD.BluRay.DV.HDR.x265-GRP.mkv  26.3 GB
   films/Nosferatu.1922.1080p.BluRay.x264.mkv  8.1 GB
   Type the path to a file  or a mount point
  ↑↓ move · 1-9 jump · enter choose · q abort
```

A typed directory is descended into rather than refused, which is how a library too big to
list is still navigable. With no terminal there is nobody to ask, so it exits 2 with the
shape of the command instead.

### What the filename is worth

Only a guess, and the `Name` line says what was made of it: the **last** year-shaped token
wins (`Blade.Runner.2049.2017.2160p` → *Blade Runner 2049*, 2017), and a name with nothing
in it (`movie.mkv`, `title00.mkv`) falls back to the directory it sits in, which is how
`Blade Runner (1982)/movie.mkv` still finds the film. That year is also the last thing the
grain estimate has to go on when no film was identified.

### What it writes

Two files, named after the film, next to it — or in `--outdir`:

| File | For |
| ---- | --- |
| `<name>.graindamage.json` | HandBrake: *Presets → Import from file*. One document holding **both** presets, so AV1 and x265 both appear in the list and you pick per queue item. |
| `<name>.graindamage.sh` | FFmpeg: already `chmod +x`. The first plan runs; the other sits commented out directly beneath it, so switching encoder is deleting one `#`. |

The script `cd`s to the film's directory and names the film by its basename, so one written
inside a container still runs on the host that mounted it. Its header carries the film, the
grain level and its reasons, and where the technical rows came from.

### Flags

Every one of them is optional, including the file. They exist to skip a step you already
know the answer to — and, with `--no-menu` or `--yes`, to answer all of them at once from a
script.

| Flag | Does |
| ---- | ---- |
| `--title TITLE` | Search for this instead of what the filename says |
| `--year YEAR` | The release year, when the name has none |
| `--imdb-id ttNNNNNNN` | Use this title id for the technical rows; enough on its own, with no TMDB key |
| `--specs FILE` | Read the technical specifications from a file instead of asking Gemini |
| `--encoder svt-av1\|x265` | Which encoder leads: the live command, and the first preset |
| `--speed quality\|balanced\|fast` | How much CPU time you will spend |
| `--size archival\|balanced\|compact` | Where to sit on the size/fidelity curve |
| `--grain none\|light\|moderate\|heavy\|extreme` | State the grain instead of letting it be inferred |
| `--bit-depth 8\|10\|12` | Force the encoding bit depth |
| `--outdir DIR` | Write the two files here instead |
| `--force` | Replace files that are already there |
| `--no-gemini` | No Gemini at all: no review, no technical look-up |
| `--no-menu` | Take the answers from these flags rather than asking step by step |
| `-y`, `--yes` | No questions at all: the best-matching hit, no menus |
| `--ffprobe PATH` | A non-default `ffprobe` |
| `-q`, `--quiet` | Print only the two paths that were written |

Exit codes:

| Exit | Means |
| ---- | ----- |
| `0` | Both files written |
| `1` | You stopped it; nothing was written |
| `2` | Setup or usage: no such file, a directory, no file to work on, `ffprobe` missing or refusing the file, outputs already there on the flag path, a bad flag |
| `130` | Ctrl-C |

No key is required. Without `TMDB_API_KEY` the film step offers the IMDb-id row and the
film-less row instead of hits; without `GEMINI_API_KEY` there is no look-up row and no
review line, and pasting is the way to the rows. Each missing piece costs a warning in the
report, never the settings.

## Run with Docker

```bash
docker compose up --build      # http://localhost:8080
```

`compose.yaml` also mounts your films at `/media` — set `MEDIA_DIR` in `.env` to point it
at your library — so that the command line inside the container has something to work on.

Or without compose:

```bash
docker build -t graindamage .
docker run --rm -p 8080:8080 --env-file .env graindamage
```

The image runs as a non-root user and ships a `HEALTHCHECK` against `/healthz`. It also
carries `ffmpeg`, for the `ffprobe` the command line runs — which is most of its size
(roughly 850 MB, against 260 MB without). See
[Inside the container](#inside-the-container) for the two-line recipe: `docker compose
exec` into a shell, then `graindamage /media/film.mkv`.

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

False positives are listed by fingerprint in [`.gitleaksignore`](.gitleaksignore), one
per line with the reason it is safe. Prefer changing the value over adding a line: a
test that needs a credential-shaped string can build it from parts at runtime, which
keeps the scanner useful. History is a different matter — a published commit cannot be
un-flagged any other way.

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
    pipeline.py         the tail both front-ends share: review, note, commands
    validate.py         allowlist for anything a model returns
    encoders.py         HandBrake and FFmpeg commands, HandBrake presets
  cli/
    app.py              the argparse surface and the run itself
    wizard.py           the four steps, and the file picker when given none
    session.py          what both paths share: clients, rows, the two exceptions
    filename.py         Movie.Title.1982.2160p… → title and year
    probe.py            ffprobe as a subprocess, never a shell
    prompts.py          the arrow-key picker, and its numbered fallback
    report.py           the advice as terminal text
    outputs.py          the .json preset and the .sh script it writes
  providers/
    tmdb.py             title search and metadata
    imdb.py             /technical parsing: paste or look-up
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
- **IMDb technical specifications** are never scraped. Gemini is asked for them and
  says how sure it is, or you paste them from the `/technical` page yourself; a paste
  always wins.
- **Your source file** matters more than the film's release specs for CRF and bitrate,
  which is why all three source formats are parsed and every field can be overridden.

Not affiliated with IMDb. This product uses the TMDB API but is not endorsed or certified
by TMDB.
