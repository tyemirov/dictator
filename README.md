# Dictator

## Archival-voice extraction and long-form voice-cloning

Two small, self-contained Python utilities:

| Script           | Purpose                                                                                                                                                                                                                                       |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`extract.py`** | Carve out the **clearest window** (default 20 s) from a noisy archival recording using Whisper ASR confidence + SNR heuristics, with a speaker-diarization model provided by `pyannote.audio`, then output a peak-normalised 24 kHz mono WAV. |
| **`main.py`**    | Feed that reference sample (or any WAV/MP3) plus its transcript to **Qwen3-TTS** and synthesise arbitrarily long speech from plain text – again to a peak-normalised 24 kHz mono WAV.                                                     |

---

## Prerequisites

You **must** run this project under **Python 3.11.8**, as some dependencies (pyannote.audio, Qwen3-TTS) don’t ship
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

### GPU requirement

An NVIDIA GPU is required for supported Dictator operation. CPU deployment is not supported.

## Service Container

The repo includes a GPU-only containerized gRPC service runtime with the required system packages:

* Python 3.11.8
* FFmpeg
* `libsndfile`
* `espeak-ng`
* build tooling for Python wheels
* Python packages from `requirements.txt`
### Build GPU Image

```bash
docker build -f Dockerfile.gpu -t dictator:gpu .
```

### GPU Release Flow

Dictator owns one production declaration at
`.mprlab/deploy/resources.yml`. It declares the GPU image, ComputerCat service,
retained model and artifact volumes, private runtime values, versioned
`dictator.grpc` capability, and the public `dictator.mprlab.com` route. The
application contains no production controller implementation.

The complete production lifecycle is:

```bash
make release && make publish && make deploy
```

Each target resolves the physical sibling gateway at exactly
`../mprlab-gateway` and delegates to its selected-application lifecycle.
Release runs Dictator CI and seals the exact Linux AMD64 GPU image. Publication
publishes that sealed image without rebuilding. Deployment consumes the sealed
publication receipt, reconciles only Dictator-owned resources, admits no
provider source checkout, and verifies the declared runtime capability and
public route.

Deployment reads `DICTATOR_GRPC_AUTH_TOKEN` and `HF_TOKEN` only from the
ignored mode-`0600` `.mprlab/deploy/.env`. Release and publication do not read
that private file.

### Run with Compose

If you want Docker Compose to load service environment from a file, create a private local `.env` explicitly. Use `dictator.env.example` only to review variable names; its values are intentionally unusable and the file must never be copied or sourced.

```bash
install -m 0600 /dev/null .env
# edit .env and set DICTATOR_GRPC_AUTH_TOKEN
# leave HF_TOKEN as ${HF_TOKEN} so Compose reads it from your shell
export HF_TOKEN=...

docker compose up --build
```

The application itself does not read env files. It only reads process environment. `docker-compose.yml` uses Docker Compose interpolation, so `DICTATOR_GRPC_AUTH_TOKEN` can come from `.env` or your shell environment. Keep `HF_TOKEN=${HF_TOKEN}` in `.env` so the key exists there, but the actual Hugging Face token still comes from your shell environment. Compose fails fast if that exported `HF_TOKEN` is missing.

This requires a host GPU plus the NVIDIA Container Toolkit so Docker can pass the device through.

```bash
docker compose up --build dictator
```

Or with the convenience wrappers:

```bash
./scripts/up.sh
./scripts/down.sh
```

### Run the Published Dictator Image from GHCR

If you want to smoke-test the published GPU image instead of building from the checkout, use:

```bash
docker compose --profile ghcr-gpu up dictator-image
```

The published Dictator image currently lives at:

```text
ghcr.io/tyemirov/dictator:latest
```

To pin a specific release tag instead of `latest`, override `DICTATOR_IMAGE`:

```bash
DICTATOR_IMAGE=ghcr.io/tyemirov/dictator:1.2.3 \
docker compose --profile ghcr-gpu up dictator-image
```

The container starts the gRPC server with:

```bash
python serve.py --config /app/config.yml
```

For the current Python client contract, async job model, integration path, and best practices, see [docs/client-integration.md](docs/client-integration.md).

### Python client quick start

For most callers, the Python convenience clients are the right integration surface.

```python
import grpc
from pathlib import Path

from dictator.client import AlignmentClient

channel = grpc.insecure_channel("127.0.0.1:50051")
client = AlignmentClient(channel, metadata=(("x-dictator-token", "your-token"),))
result = client.align_file(
    Path("sample.wav"),
    transcript_file=Path("sample.txt"),
    language_code="en",
)
print(result.srt_artifact_id)
```

Use the blocking convenience methods by default. For explicit submit/get/wait job flows and endpoint-specific contract details, use [docs/client-integration.md](docs/client-integration.md).

### Go contract module

Dictator owns the authoritative `dictator.speech.v1` protobuf contract and
publishes generated Go bindings in
`github.com/tyemirov/dictator/sdk/go/dictatorspeechv1`.

To regenerate the checked-in Python and Go gRPC artifacts from the proto
sources, run:

```bash
make proto
```

`make proto` bootstraps a lightweight local Python codegen environment under
`tools/proto-python`, downloads a pinned `protoc` toolchain under
`tools/protoc`, and auto-installs the pinned Go protobuf generators when
needed. The main test and coverage targets still use `.venv/bin/python`
automatically when a project venv is present.

The compose setup mounts persistent caches for Hugging Face, Whisper, and Torch models, plus `.dictator-artifacts` for generated service artifacts.

### Notes

* GPU is required. Supported deployment assumes an NVIDIA GPU and the NVIDIA Container Toolkit.
* `HF_TOKEN` is required in the container environment and should be exported in your shell before starting Dictator.
* Both `Dockerfile` and `Dockerfile.gpu` use multi-stage builds so compilers and other build-only packages stay out of the final runtime image.
* `Dockerfile.gpu` installs the CUDA 12.8 Torch wheel set and is the supported runtime image.
* `Dockerfile.gpu` now prefetches the default Qwen voice-cloning model, `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, and the Silero Russian `v5_5_ru` preset-speaker model into `/opt/models` during the image build, so the synthesis engines do not need to download weights on first container startup. The Silero package is verified with `DICTATOR_SILERO_RU_MODEL_SHA256` before it is loaded.
* That `1.7B` default is the quality-first choice for voice cloning, not the smallest one. Expect a larger GPU image and a slower bake/prefetch step than with the lighter `0.6B` model.
* `Dockerfile.gpu` also installs the official `flash-attn` wheel for Torch 2.8 / CUDA 12 and the `sox` binary so the baked Qwen3 runtime has its intended acceleration and toolchain available at startup.
* The published container package is `ghcr.io/tyemirov/dictator`.
* The GPU container still uses Python 3.11.8. The host provides the actual NVIDIA device through Docker; without that runtime integration, Dictator will not function correctly.

## Browser Voice Clone Demo

The repo includes a browser example that records a short voice sample, calls the Dictator gRPC API through a small local HTTP bridge, converts the browser recording to WAV, and downloads a WAV of a selected preset passage read back from the full recorded sample in the user's cloned voice.

Run the example after Dictator is already up:

```bash
python -m demo.voice_clone_web.app --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080` and record a voice sample. The browser does not hold the Dictator auth token; the backend bridge reads `DICTATOR_GRPC_AUTH_TOKEN` from its environment.

The page asks the user to read:

> I grew up near a busy street, so even now I sleep best with a little noise in the distance.
> My friends know I speak quickly when I am excited, slow down when I am serious, and laugh before I finish the punch line.
> On cold mornings I want strong coffee, warm light, and ten quiet minutes to think.
> If you know me well, you can hear the difference between my polite voice, my tired voice, and the one I use when I am truly delighted.

The browser page no longer chooses the Dictator gRPC target. The backend bridge owns that connection and, in Docker, always talks to the internal alias `dictator-grpc:50051`.
The demo now uses Qwen3-TTS only, so the full recorded sample plus its fixed transcript go directly into the voice-cloning request.

For a Dockerized test page using `ghttp` as the frontend and the existing Python HTTP bridge behind `/api`, provide a local TLS certificate and key and start:

```bash
TLS_CERT_HOST_PATH=/path/to/computercat-cert.pem \
TLS_KEY_HOST_PATH=/path/to/computercat-key.pem \
docker compose --profile voice-clone-demo up -d voice-clone-web voice-clone-bridge
```

Or use the wrapper:

```bash
./scripts/voice-clone-demo.sh \
  --tls-cert /path/to/computercat-cert.pem \
  --tls-key /path/to/computercat-key.pem
```

The wrapper is the easier path for the demo-only stack. It still supplies a harmless placeholder `HF_TOKEN` when you are only starting `voice-clone-web` + `voice-clone-bridge`, but the bridge now requires a real `DICTATOR_GRPC_AUTH_TOKEN` because it authenticates to Dictator on the browser's behalf.
It also falls back to the repo-root `.env` for `TLS_CERT_HOST_PATH` and `TLS_KEY_HOST_PATH`, so once those are set there you can run `./scripts/voice-clone-demo.sh` directly.
The bridge always connects to the internal Docker alias `dictator-grpc:50051`. The Dictator service profiles expose that shared alias, so the demo talks to one stable hostname whether you run `dictator` or `dictator-image`.
That means the demo-only wrapper expects a Dictator container to already be running in the same Compose project and network. If you want the wrapper to start Dictator too, use `--with-dictator`.

Stop it with:

```bash
./scripts/voice-clone-demo-down.sh
```

By default the demo stack publishes on port `8001`. The wrapper prints the public URL after startup as `https://computercat.tyemirov.net:8001/` unless you override the host or port with `--host` / `--port` or `VOICE_CLONE_DEMO_HOST` / `VOICE_CLONE_WEB_PORT`. The wrapper accepts the TLS files through `--tls-cert` / `--tls-key` or `TLS_CERT_HOST_PATH` / `TLS_KEY_HOST_PATH`.

If Dictator is also running through the GHCR GPU stack on the same machine, you can launch both together:

```bash
TLS_CERT_HOST_PATH=/path/to/computercat-cert.pem \
TLS_KEY_HOST_PATH=/path/to/computercat-key.pem \
docker compose --profile ghcr-gpu --profile voice-clone-demo up -d dictator-image voice-clone-web voice-clone-bridge
```

With the wrapper:

```bash
./scripts/voice-clone-demo.sh \
  --with-dictator \
  --tls-cert /path/to/computercat-cert.pem \
  --tls-key /path/to/computercat-key.pem
```

And stop both with:

```bash
./scripts/voice-clone-demo-down.sh --with-dictator
```

The `ghttp` service proxies `/api/clone` to the local bridge container, so the browser stays same-origin while the bridge connects to Dictator over the internal Docker hostname `dictator-grpc:50051`.

Example-specific notes live in [demo/voice_clone_web/README.md](./demo/voice_clone_web/README.md).

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
  --sample-text TEXT    reference transcript for the sample audio
  --speech JSON        write JSON timeline with text/metadata
  --force                overwrite existing output
```

* Input text is **cleaned** (Unicode NFKC, whitespace collapsed).
* Voice cloning now uses **Qwen3-TTS** only.
* The default model is `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
* Qwen3-TTS uses the full speaker sample plus its transcript and packs sentences by tokenizer budget.
* The gRPC synthesis service defaults `language_code=ru` requests with no explicit engine and no reference-speaker fields to **Silero `v5_5_ru`** at 24 kHz, using preset speaker `baya` unless `preset_speaker` is set to another discovered Silero speaker (`aidar`, `baya`, `kseniya`, `eugene`, or `xenia`); callers can discover Silero preset speakers with `ListSynthesisVoices`, request another positive output sample rate with `audio_format.sample_rate_hz`, and pass Silero-only SSML via `text_format=SYNTHESIS_TEXT_FORMAT_SSML` for `<break>`, `<prosody>`, `<p>`, and `<s>` controls.
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
    * qwen-tts

---

## License

This project is proprietary software. All rights reserved by Marco Polo Research Lab.

Qwen3-TTS, Whisper, and Silero licenses apply to their respective models. Review Silero model licensing before commercial use.

See the [LICENSE](./LICENSE) file for details.
