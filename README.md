# Dictator

## Archival-voice extraction and long-form voice-cloning

Two small, self-contained Python utilities:

| Script           | Purpose                                                                                                                                                                                                                                       |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`extract.py`** | Carve out the **clearest window** (default 20 s) from a noisy archival recording using Whisper ASR confidence + SNR heuristics, with a speaker-diarization model provided by `pyannote.audio`, then output a peak-normalised 24 kHz mono WAV. |
| **`main.py`**    | Feed that reference sample (or any WAV/MP3) to **[XTTS-v2]** and synthesise arbitrarily long speech from plain text – again to a peak-normalised 24 kHz mono WAV.                                                                             |

---

## Prerequisites

You **must** run this project under **Python 3.11.8**, as some dependencies (pyannote.audio, Coqui-TTS) don’t ship
wheels for newer interpreters. We recommend installing via **pyenv**:

```bash
# 1) Install pyenv (if needed)
curl https://pyenv.run | bash
# Add these to your shell startup (~/.bashrc or similar):
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# 2) Install Python 3.11.8
pyenv install 3.11.8

# 3) Use 3.11.8 in this project directory
cd /path/to/dictator
pyenv local 3.11.8

# 4) Create & activate your venv
python3.11 -m venv .venv
source .venv/bin/activate

# 5) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
````

---

## Quick start (GPU)

```bash
# 🔈 1) cut a 20-second Churchill clip
python extract.py \
       --input  assets/sources/voices/churchill_disc1side2.flac \
       --output assets/samples/churchill.wav

# 📖 2) clone the voice and read Genesis, capped at 3 minutes
python main.py \
       --sample assets/samples/churchill.wav \
       --text   assets/sources/texts/genesis.txt \
       --output output/genesis_by_churchill.wav \
       --length 3m
```

### CPU-only?

Both scripts fall back to `"cpu"` if CUDA is missing. `main.py` on CPU will be **slow** – consider shorter texts.

## Service Container

The repo now includes a containerized gRPC service runtime with the required system packages:

* Python 3.11.8
* FFmpeg
* `libsndfile`
* `espeak-ng`
* build tooling for Python wheels
* Python packages from `requirements.txt`

### Build

```bash
docker build -t dictator:local .
```

### Build GPU Image

```bash
docker build -f Dockerfile.gpu -t dictator:gpu .
```

### GPU Release Flow

Pushing a semver Git tag like `1.2.3` or `v1.2.3` now triggers a GitHub Actions validation workflow that:

* verifies the tagged commit is contained in `master`
* reruns the unit test workflow on that tagged commit

The actual GPU image build and push now happens locally so Buildx layer cache can be reused across releases.

To publish the GPU image for a checked-out tag:

```bash
git fetch --tags origin
git checkout v1.2.3
./scripts/docker-gh-deploy.sh v1.2.3
```

Or through `make`:

```bash
make publish-gpu-image TAG=v1.2.3
```

The publish script:

* requires a clean working tree
* verifies the checked-out tag resolves to a commit contained in `origin/master`
* runs `make ci` before publishing
* reuses a persistent local Buildx cache in `.buildx-cache-gpu`
* derives the default GHCR image as `ghcr.io/<owner>/<repo>-gpu`

For a stable release tag like `v1.2.3`, it publishes:

* `:1.2.3`
* `:1.2`
* `:1`
* `:latest`
* `:v1.2.3`

For prerelease tags like `v1.2.3-rc.1`, it only publishes the exact prerelease tags and does not move `:latest`.

If you are not already logged into `ghcr.io`, either run `docker login ghcr.io` first or set `GHCR_USERNAME` and `GHCR_TOKEN` before running the script.

### Run with Compose

If you want Docker Compose to load service environment from a file, create a local `.env` from `dictator.env.example`, then start the service:

```bash
cp dictator.env.example .env
# edit .env and set DICTATOR_GRPC_AUTH_TOKEN
# set HF_TOKEN as well if you want diarization / speaker extraction

docker compose up --build
```

The application itself does not read env files. It only reads process environment. `docker-compose.yml` uses Docker Compose interpolation, so `DICTATOR_GRPC_AUTH_TOKEN` and `HF_TOKEN` can come from `.env` or from your shell environment.

### Run with Compose on CUDA

This requires a host GPU plus the NVIDIA Container Toolkit so Docker can pass the device through.

```bash
docker compose --profile gpu-local up --build dictator-gpu
```

### Run the Published GPU Image from GHCR

If you want local orchestration to always pull the released container instead of building from the checkout, use:

```bash
docker compose --profile ghcr-gpu up dictator-ghcr
```

Or with the convenience wrappers:

```bash
./scripts/up.sh
./scripts/down.sh
```

The GHCR GPU service profile uses `pull_policy: always` and defaults to:

```text
ghcr.io/tyemirov/dictator-gpu:latest
```

To pin a specific release tag instead of `latest`, override `DICTATOR_IMAGE`:

```bash
DICTATOR_IMAGE=ghcr.io/tyemirov/dictator-gpu:1.2.3 \
docker compose --profile ghcr-gpu up dictator-ghcr
```

The container starts the gRPC server with:

```bash
python serve.py --config /app/config.yml
```

The compose setup mounts persistent caches for Hugging Face, Whisper, and Torch models, plus `.dictator-artifacts` for generated service artifacts.

### Notes

* The provided image is CPU-oriented. It includes the service prerequisites and will start the server, but heavy transcription/alignment/TTS workloads will be slow without GPU acceleration.
* The container installs CPU `torch` / `torchaudio` wheels explicitly so it does not pull CUDA runtimes into a CPU deployment.
* `HF_TOKEN` should be present in the container environment, via `.env` or your shell environment, if you want pyannote diarization and archival speaker extraction to download gated Hugging Face models on first run.
* Both `Dockerfile` and `Dockerfile.gpu` use multi-stage builds so compilers and other build-only packages stay out of the final runtime image.
* `Dockerfile.gpu` installs the CUDA 12.8 Torch wheel set and is intended to run with the `gpu-local` profile in `docker-compose.yml`.
* The GPU container still uses Python 3.11.8. The host provides the actual NVIDIA device through Docker; without that runtime integration, the image will build but CUDA execution will not be available.

## Browser Voice Clone Demo

The repo includes a browser example that records a short voice sample, calls the Dictator gRPC API through a small local HTTP bridge, and downloads a WAV of Genesis 1:1-10 read back in the user's cloned voice.

Run the example after Dictator is already up:

```bash
python -m examples.voice_clone_web.app --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080` and provide:

* `Dictator gRPC URL`, for example `localhost:50051` or `https://your-host:443`
* `Auth Token`, matching `DICTATOR_GRPC_AUTH_TOKEN`
* `Language Code`, usually `en`

The page asks the user to read:

> The quick brown fox jumped over the lazy dog. Eleven benevolent elephants balanced on bright blue bicycles.

When the page is served from a non-local hostname, the `Dictator gRPC URL` field auto-fills to `<current-host>:50051`. So if you publish the page at `computercat.tyemirov.net`, it defaults to `computercat.tyemirov.net:50051`.

For a Dockerized test page using `ghttp` as the frontend and the existing Python HTTP bridge behind `/api`, start:

```bash
docker compose --profile voice-clone-demo up -d voice-clone-web voice-clone-bridge
```

Or use the wrapper:

```bash
./scripts/voice-clone-demo.sh
```

Stop it with:

```bash
./scripts/voice-clone-demo-down.sh
```

By default the demo stack publishes on port `8001`. The wrapper prints the public URL after startup and defaults that hostname to `computercat.tyemirov.net`. Override either value with `--host` / `--port` or `VOICE_CLONE_DEMO_HOST` / `VOICE_CLONE_WEB_PORT`.

If Dictator is also running through the GHCR GPU stack on the same machine, you can launch both together:

```bash
docker compose --profile ghcr-gpu --profile voice-clone-demo up -d dictator-ghcr voice-clone-web voice-clone-bridge
```

With the wrapper:

```bash
./scripts/voice-clone-demo.sh --with-dictator
```

And stop both with:

```bash
./scripts/voice-clone-demo-down.sh --with-dictator
```

The `ghttp` service proxies `/api/clone` to the local bridge container, so the browser stays same-origin while the bridge connects to the Dictator gRPC endpoint named in the page.

Example-specific notes live in [examples/voice_clone_web/README.md](./examples/voice_clone_web/README.md).

---

## Folder layout

```text
dictator/
├── assets/
│   ├── sources/
│   │   ├── voices/      # original long recordings
│   │   └── texts/       # .txt books / speeches / etc.
│   └── samples/         # reference clips cut by extract.py
├── output/              # final generated audio
├── extract.py
├── main.py
├── requirements.txt
└── README.md            # ← you are here
```

Feel free to change folders – the scripts just take full paths.

---

## `extract.py` usage

```text
usage: extract.py --input FILE --output FILE [options]

optional arguments
  --model {tiny,base,small,medium,large-v2,large-v3}
                        Whisper size (default: medium)
  --duration SECONDS    window length (default: 20)
  --min-confidence P    keep words whose P ≥ threshold (default: 0.80)
  --language CODE       Whisper language (e.g. 'en'); auto-detect if omitted
  --max-speech-rate R   discard windows faster than R words/s (default: 4)
  --min-centroid HZ     discard windows below this spectral centroid
                         (default: 500)
  --max-centroid HZ     discard windows above this spectral centroid
                         (default: 4000)
  --timeouts D T R      seconds for decode / transcribe / trim
  --force               overwrite existing output
```

**Algorithm**

1. FFmpeg → mono 16 kHz PCM
2. Whisper full-track transcription (progress heartbeat every 5 %)
3. Slide a fixed window; discard windows with spectral centroids outside
   `[MIN_CENTROID_HZ, MAX_CENTROID_HZ]`, then rank by
   *(word count × avg confidence × SNR × (1 + variation))*
4. FFmpeg lossless trim + peak-normalise to –1 dBFS, resample 24 kHz

Typical runtime on an RTX 3060 for a 30-minute 44 kHz FLAC is \~70 s.

---

## `main.py` usage

```text
usage: main.py --sample WAV/MP3 --text TXT --output WAV [options]

optional arguments
  --length 10s|3m|1.5h   cap final audio; stops on last full sentence
  --language CODE       TTS language code (default: en)
  --speech JSON        write JSON timeline with text/metadata
  --force                overwrite existing output
```

* Input text is **cleaned** (Unicode NFKC, whitespace collapsed).
* **Smart-split** into ≤ 800 bytes so XTTS never truncates mid-chunk.
* Synthesis stops when the next sentence would exceed `--length`.
* All chunks concatenated with FFmpeg, `dynaudnorm` + –1 dBFS, 24 kHz mono.
* When `--speech` is provided, a JSON file is written containing:

  ```json
  {
    "textSegments": [{"start": 0.0, "end": 4.1, "content": "Once upon a time"}],
    "imageCues": [],
    "voices": [{"id": "sample", "label": "sample", "file": "voice.wav"}]
  }
  ```

---

## Dependencies

* **Python 3.11.8** (via pyenv; see “0 Prerequisites”)
* **FFmpeg** (with `dynaudnorm` filter) – e.g. `sudo apt install ffmpeg`
* Python libraries in `requirements.txt`

    * Torch wheels matching your CUDA version (see the [PyTorch site](https://pytorch.org/get-started/locally/))
    * ffmpeg-python
    * soundfile
    * numpy
    * openai-whisper
    * pyannote.audio
    * coqui-tts

---

## License

This project is proprietary software. All rights reserved by Marco Polo Research Lab.

XTTS-v2 and Whisper licenses apply to their respective models.

See the [LICENSE](./LICENSE) file for details.
