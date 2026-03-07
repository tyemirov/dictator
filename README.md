# Dictator

Dictator is a Python 3.11.8 speech execution service with a gRPC API for:

- audio transcription
- structured speaker diarization
- forced alignment
- grouped subtitle rendering
- reference voice extraction
- long-form voice-cloned speech synthesis

The repo still contains local compatibility CLIs for the earlier script workflows, but the primary shape now is a token-protected gRPC service.

## Current Capabilities

| Capability | What it does |
| --- | --- |
| Transcription | Converts uploaded audio into text, with optional word-level timestamps |
| Diarization | Returns structured JSON with request-local speakers (`S1`, `S2`, ...), utterances, words, and speaker segments |
| Forced alignment | Aligns known source text against audio and returns aligned words plus SRT |
| Subtitle rendering | Builds grouped SRT cues by `N` words or `N` sentences; with `source_text`, this uses forced alignment |
| Reference extraction | Extracts a clean speaker reference sample from noisy archival material |
| Speech synthesis | Generates cloned speech from text using an extracted or supplied speaker reference |
| Artifact storage | Uploads and downloads audio, text, JSON, and SRT artifacts |
| Runtime metrics | Exposes request, latency, inflight, and byte counters |

## API Surface

The server currently exposes these gRPC services under [`proto/dictator/speech/v1`](/home/tyemirov/Development/tyemirov/dictator/proto/dictator/speech/v1):

| Service | RPCs |
| --- | --- |
| `ArtifactService` | `UploadArtifact`, `DownloadArtifact` |
| `TranscriptionService` | `Transcribe`, `DiarizeAudio` |
| `AlignmentService` | `AlignTranscript` |
| `SubtitleService` | `RenderSubtitles` |
| `VoiceService` | `ExtractReferenceSample`, `SynthesizeSpeech` |
| `RuntimeService` | `GetMetrics` |

There is also a standard gRPC health service registered by the server.

## Request Model

Most workflows follow the same pattern:

1. Upload audio or text with `ArtifactService.UploadArtifact`
2. Call a speech RPC with the returned `artifact_id`
3. Read inline fields from the RPC response
4. Optionally download generated artifacts such as SRT, WAV, timeline JSON, or diarization JSON

This keeps the service independent of the caller's local filesystem.

## Auth and Transport

- The gRPC service is protected by a token
- Callers authenticate with either:
  - `x-dictator-token: <token>`
  - `Authorization: Bearer <token>`
- By default the server listens with insecure gRPC transport and token auth
- TLS is not configured in this repo yet

The token is typically provided through:

- [`config.yml`](/home/tyemirov/Development/tyemirov/dictator/config.yml)
- [`.env.example`](/home/tyemirov/Development/tyemirov/dictator/.env.example)

Example:

```dotenv
DICTATOR_GRPC_AUTH_TOKEN=replace-with-a-long-random-token
HF_TOKEN=
```

`HF_TOKEN` is needed for pyannote-backed diarization / reference extraction model downloads.

## Local Development Prerequisites

You should run the project with Python `3.11.8`.

System dependencies for local non-container runs:

- `ffmpeg`
- `libsndfile`
- `espeak-ng`
- build tooling for Python wheels

Suggested setup:

```bash
pyenv install 3.11.8
pyenv local 3.11.8

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Local Service Quick Start

1. Create the environment file:

```bash
cp .env.example .env
```

2. Set at least:

```dotenv
DICTATOR_GRPC_AUTH_TOKEN=replace-with-a-long-random-token
```

3. Start the server:

```bash
python serve.py --config config.yml --env-file .env
```

By default this starts the gRPC service on `0.0.0.0:50051`.

## Client and CLI Helpers

Thin client helpers live under [`dictator/client`](/home/tyemirov/Development/tyemirov/dictator/dictator/client):

- `DictationClient`
- `DiarizationClient`
- `SubtitleClient`

Current CLI entrypoints:

| File | Purpose |
| --- | --- |
| [`serve.py`](/home/tyemirov/Development/tyemirov/dictator/serve.py) | run the gRPC server |
| [`dictate.py`](/home/tyemirov/Development/tyemirov/dictator/dictate.py) | upload audio and print dictation JSON |
| [`subtitle.py`](/home/tyemirov/Development/tyemirov/dictator/subtitle.py) | upload audio and print or save grouped SRT |
| [`align.py`](/home/tyemirov/Development/tyemirov/dictator/align.py) | local direct forced-alignment CLI over the packaged service |
| [`extract.py`](/home/tyemirov/Development/tyemirov/dictator/extract.py) | local direct reference extraction CLI |
| [`main.py`](/home/tyemirov/Development/tyemirov/dictator/main.py) | local direct synthesis CLI |

### Dictation Example

```bash
python dictate.py \
  --config config.yml \
  --env-file .env \
  --input sample.wav \
  --autodetect-language
```

This prints:

```json
{"text": "hello world"}
```

### Grouped Subtitle Example

Transcription mode:

```bash
python subtitle.py \
  --config config.yml \
  --env-file .env \
  --input sample.wav \
  --autodetect-language \
  --granularity words \
  --group-size 2 \
  --output sample.srt
```

Forced-alignment mode:

```bash
python subtitle.py \
  --config config.yml \
  --env-file .env \
  --input sample.wav \
  --language en \
  --granularity sentences \
  --group-size 1 \
  --source-text-file transcript.txt \
  --output sample.srt
```

Behavior:

- if `source_text` is absent, subtitles come from transcription
- if `source_text` is present, subtitles come from forced alignment
- `granularity=words` groups every `N` timed words
- `granularity=sentences` groups every `N` timed sentence units

## Diarization Output

`DiarizeAudio` returns a structured JSON object via `google.protobuf.Struct`.

Directionally, the payload can include:

- `text`
- `languageCode`
- `speakers`
- `utterances`
- `words`
- `speakerSegments`

Speaker labels are request-local only. `S1` in one request is not a persistent speaker identity across files.

## Voice Workflows

The voice pipeline currently supports:

1. upload long/noisy source audio
2. call `ExtractReferenceSample`
3. upload or inline text
4. call `SynthesizeSpeech`

`SynthesizeSpeech` can optionally return inline timeline segments and persist a timeline JSON artifact.

## Docker

### CPU Image

```bash
docker build -t dictator:local .
docker compose up --build
```

### CUDA Image

```bash
docker build -f Dockerfile.gpu -t dictator:gpu .
docker compose -f compose.yml -f compose.gpu.yml up --build
```

Notes:

- [`Dockerfile`](/home/tyemirov/Development/tyemirov/dictator/Dockerfile) installs CPU `torch` / `torchaudio`
- [`Dockerfile.gpu`](/home/tyemirov/Development/tyemirov/dictator/Dockerfile.gpu) installs CUDA 12.8 wheels
- [`compose.yml`](/home/tyemirov/Development/tyemirov/dictator/compose.yml) mounts `.env`, `config.yml`, model caches, and artifact storage
- [`compose.gpu.yml`](/home/tyemirov/Development/tyemirov/dictator/compose.gpu.yml) requests `gpus: all`

## Configuration

Primary server configuration is in [`config.yml`](/home/tyemirov/Development/tyemirov/dictator/config.yml):

```yaml
grpc:
  host: 0.0.0.0
  port: 50051
  max_workers: 4
  max_message_bytes: 67108864
  max_inflight: 4
  download_chunk_bytes: 1048576
  artifact_root: .dictator-artifacts
  auth_token: ${DICTATOR_GRPC_AUTH_TOKEN}
```

The service resolves `${ENV_VAR}` placeholders from `.env` and the process environment.

## Validation

Local developer entrypoints:

```bash
make test
make coverage
make ci
```

`make ci` runs the coverage-enforced test suite.

This repo currently enforces `100%` line coverage on production Python code.

## Package Layout

The live service code is organized under [`dictator/`](/home/tyemirov/Development/tyemirov/dictator/dictator):

```text
dictator/
├── alignment/          # forced alignment logic and WhisperX backend
├── audio/              # ffmpeg-based audio helpers
├── client/             # thin gRPC client helpers
├── diarization/        # speaker assignment and JSON shaping
├── extraction/         # reference sample extraction
├── runtime/            # errors, metrics, inflight, model runtime
├── speech/v1/          # generated protobuf/grpc stubs
├── storage/            # local artifact store
├── subtitles/          # grouped subtitle rendering
├── synthesis/          # XTTS-backed speech synthesis
├── transcription/      # Whisper-backed transcription
└── transport/grpc/     # gRPC config, server, and servicers
```

## Legacy / Local Utility Workflows

The earlier local script workflows still work:

- [`extract.py`](/home/tyemirov/Development/tyemirov/dictator/extract.py): extract a strong reference sample from long audio
- [`main.py`](/home/tyemirov/Development/tyemirov/dictator/main.py): synthesize long-form speech from a reference sample and text
- [`align.py`](/home/tyemirov/Development/tyemirov/dictator/align.py): align a transcript to audio and emit SRT

Those remain useful for direct local use, but they are no longer the best description of the repo.

## License

This project is proprietary software. All rights reserved by Marco Polo Research Lab.

XTTS-v2 and Whisper licenses apply to their respective models.

See [LICENSE](./LICENSE) for details.
