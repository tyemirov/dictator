# Dictator gRPC Service Plan

## Goal

Turn the current repository from two local scripts into a long-running service that exposes the same capabilities over gRPC:

1. Extract a clean reference voice sample from a longer recording.
2. Synthesize long-form speech from text using a reference sample.

The plan below is based on the current code in `extract.py`, `main.py`, `whisper_service.py`, and the existing tests under `tests/`.

## Current Codebase Summary

### What exists today

- `extract.py`
  - Decodes an input recording with FFmpeg.
  - Runs Whisper transcription.
  - Runs pyannote speaker diarization.
  - Chooses the best fixed-duration window.
  - Trims and normalizes the chosen sample to 24 kHz mono WAV.
- `main.py`
  - Cleans text.
  - Splits text into XTTS-safe chunks.
  - Loads XTTS-v2 and synthesizes one WAV per chunk.
  - Concatenates and normalizes the final output.
  - Optionally retranscribes the final output to produce a word timeline JSON.
- `whisper_service.py`
  - Loads Whisper and exposes word-level transcription helpers.
- `duration.py`
  - Parses human-readable durations for the extractor CLI.

### Useful properties to preserve

- The actual audio-processing logic is already factored into reusable functions in several places:
  - `main.py`: `clean`, `build_chunks`, `parse_length`, `synthesise`, `concat_normalise`
  - `extract.py`: `decode_pcm`, `choose_window`, `apply_diarization_filter`
  - `whisper_service.py`: `load_whisper_model`, `transcribe_words`
- The tests already cover a small set of pure behaviors:
  - duration parsing
  - XTTS chunk splitting
  - empty-audio handling
  - word extraction shape
  - synthesis timeline behavior
  - window-selection logic

## Service Conversion Constraints Found In The Current Code

### 1. The public contract is file-path based

Both scripts expect local files for input and output. That works for a CLI, but not for a network service. A service needs one of these instead:

- streamed binary uploads/downloads
- object storage references
- a managed local artifact store with IDs

### 2. Model lifecycle is request-coupled

- `main.py` loads XTTS inside `synthesise`
- `whisper_service.py` loads Whisper on demand
- `extract.py` loads the diarization pipeline inside the request flow

That is too expensive for a service. Models should be loaded once per worker process and reused across requests.

### 3. The code is not request-safe yet

- `main.py` always uses `_tts_chunks/` as a shared temp directory
- both CLIs prompt with `input(...)` before overwriting files
- logging is configured globally inside the scripts

These patterns will break under concurrent RPC traffic.

### 4. Timeout handling is CLI-oriented

`extract.py` uses `signal.alarm`, which is process-global and not suitable for a threaded or async server request model.

### 5. Import-time global state will leak into a server

`extract.py` changes torch backend flags and warning filters during import. That is acceptable for a local script, but a service should move process configuration into explicit startup code so it is deterministic and testable.

### 6. Model/cache paths are host-assumption based

`whisper_service.py` downloads or loads Whisper models under `Path.home() / ".cache" / "whisper"`. In a service deployment that location should be configurable so container, VM, and local development setups behave predictably.

### 7. Long-running GPU work needs admission control

Both major flows are heavy:

- extraction runs transcription plus diarization over the full source audio
- synthesis loads or uses XTTS and may generate many chunks for long text

A service must control how many jobs run at once, especially on a single GPU.

### 8. The current test setup is not service-ready

The test suite uses import stubbing to avoid heavy dependencies, which is useful, but it also shows the main gap: model backends and file/FFmpeg operations are not injected behind interfaces yet.

## Recommended Target Architecture

### Service shape

Build a shared speech service with four execution domains:

1. `TranscriptionService`
2. `AlignmentService`
3. `ReferenceExtractionService`
4. `SpeechSynthesisService`

Expose them through a gRPC API. Keep the CLI entrypoints as thin wrappers around the new library layer so the current workflow still works.

### Recommended package split

Introduce a real Python package and move the current script logic into service-oriented modules.

Suggested layout:

```text
src/dictator/
  app/
    config.py
    logging.py
  audio/
    ffmpeg_ops.py
    normalization.py
    tempfiles.py
  extraction/
    models.py
    service.py
    scoring.py
  synthesis/
    models.py
    service.py
    text.py
  transcription/
    models.py
    service.py
    whisper_backend.py
  alignment/
    models.py
    service.py
    whisperx_backend.py
    srt.py
  diarization/
    pyannote_backend.py
  storage/
    artifacts.py
    local_store.py
  grpc/
    server.py
    interceptors.py
    mappers.py
    proto/
cli/
  extract.py
  synthesize.py
```

The exact directory names can change, but the separation matters:

- domain logic should not know about argparse
- gRPC handlers should not know about FFmpeg command details
- model wrappers should be injected, cached, and testable

### Core abstractions to introduce

- `ArtifactStore`
  - save input audio
  - save synthesized output
  - save optional timeline JSON
  - return stable artifact IDs
- `WhisperBackend`
  - `transcribe_words(...)`
- `AlignmentBackend`
  - `align_transcript(...)`
- `DiarizationBackend`
  - `annotate_speakers(...)`
- `TtsBackend`
  - `synthesize_chunk(...)`
- `JobLimiter`
  - per-model concurrency control
  - especially important for GPU use

### Runtime model

Recommended initial deployment model:

- one long-running Python gRPC server process per GPU
- lazy-load Whisper, WhisperX alignment models, pyannote, and XTTS on first use
- keep models warm in memory
- serialize XTTS work per GPU unless profiling proves safe parallelism
- store artifacts on local disk first, with a storage abstraction so object storage can be added later
- support split deployment modes so alignment can be isolated from transcription/voice if needed

## Recommended gRPC API

### Design choice

Do not put large audio payloads directly into ordinary unary request/response messages as the primary interface. The input recordings in this repository are much larger than the default gRPC message limits, and synthesis output can also be large.

Use:

1. streaming RPCs for artifact upload/download
2. unary business RPCs that reference uploaded artifacts by ID

### Initial voice-only v1 service surface

This is the smallest Dictator-first rollout focused on the capabilities already present in this repository. The broader multi-capability speech platform surface is defined later in this document.

```proto
service ArtifactService {
  rpc UploadArtifact(stream UploadArtifactChunk) returns (UploadArtifactResponse);
  rpc DownloadArtifact(DownloadArtifactRequest) returns (stream DownloadArtifactChunk);
}

service DictatorService {
  rpc ExtractReferenceSample(ExtractReferenceSampleRequest)
      returns (ExtractReferenceSampleResponse);

  rpc SynthesizeSpeech(SynthesizeSpeechRequest)
      returns (SynthesizeSpeechResponse);
}
```

### Core request/response shapes

`ExtractReferenceSampleRequest`

- `source_artifact_id`
- `model_size`
- `language`
- `duration_seconds`
- `max_speech_rate`
- `min_centroid_hz`
- `max_centroid_hz`

`ExtractReferenceSampleResponse`

- `sample_artifact_id`
- `sample_duration_seconds`
- `trim_start_seconds`
- `trim_end_seconds`
- `detected_language` if available later

`SynthesizeSpeechRequest`

- `speaker_artifact_id`
- `text`
- `language_code`
- `max_duration_seconds`
- `include_timeline`

`SynthesizeSpeechResponse`

- `audio_artifact_id`
- `audio_duration_seconds`
- `timeline_artifact_id` or inline structured timeline for smaller outputs
- chunk count and basic metadata

### Why this API fits the current code

- It maps directly to the two existing scripts.
- It avoids file-path arguments crossing the network boundary.
- It keeps large binary transfer separate from model execution.
- It allows the first service version to stay synchronous while remaining usable.

## Cross-Repo Contract Matching

The original plan above is enough to turn Dictator into a gRPC voice service. After comparing it to `llm-proxy` and `Sheet2Tube`, the better long-term shape is a broader speech platform with four execution capabilities:

1. transcription
2. forced alignment
3. reference extraction
4. voice synthesis

### `llm-proxy` `/dictate` contract

`llm-proxy` exposes a narrow HTTP contract:

- `POST /dictate?key=...&model=...`
- `multipart/form-data`
- `audio` file field, with `file` as an alias
- success payload `{ "text": "..." }`
- error mapping:
  - `400` invalid multipart or missing file
  - `403` invalid shared secret
  - `502` upstream/transcription failure
  - `504` timeout

Current Dictator can support the core capability, but not the contract as-is:

- it already has Whisper-backed transcription primitives
- it does not have an HTTP surface
- it does not have shared-secret auth
- it does not currently return a plain `{ "text": "..." }` transcription result
- it does not use OpenAI model identifiers

Conclusion:

- the `llm-proxy` wire contract can be preserved
- the best fit is to keep `llm-proxy` as the HTTP compatibility facade
- `llm-proxy` should call a new internal speech-service transcription RPC instead of calling OpenAI directly when local transcription is desired

Recommended `llm-proxy` refactor:

- introduce a `DictationProvider` interface behind `/dictate`
- keep the existing OpenAI implementation
- add a gRPC-backed implementation that calls the central speech service
- preserve the current HTTP request/response/auth contract at the edge

This avoids pushing proxy-specific concerns into the speech runtime.

### `Sheet2Tube` speech alignment service contract

`Sheet2Tube` already has a more mature Python gRPC service, but it solves a different problem:

- input: streamed WAV plus an existing transcript
- output: aligned words plus generated SRT
- transport: gRPC streaming
- operations:
  - `Align(stream AlignChunk) returns (AlignResponse)`
  - `GetStats(StatsRequest) returns (StatsResponse)`
- operational features:
  - auth token support
  - health service registration
  - request metrics
  - deadline handling
  - inflight concurrency limits
  - model prewarm

This is not free transcription. It is forced alignment.

Overlap with Dictator:

- audio handling
- word-level timing output
- GPU/torch runtime concerns
- gRPC service delivery
- model caching and warmup concerns

Differences from Dictator:

- alignment requires a known transcript
- alignment returns SRT, not synthesized audio
- alignment uses `text/start_seconds/end_seconds`, while Dictator currently uses `content/start/end`
- `Sheet2Tube` keeps HTTP job orchestration, persistence, SSE, and artifact serving in Go
- the alignment runtime depends on `whisperx`, while Dictator uses `openai-whisper`, `pyannote.audio`, and `coqui-tts`

Conclusion:

- the alignment execution capability belongs in the same logical speech platform
- the `Sheet2Tube` HTTP job orchestration does not
- the existing `AudioToText` gRPC contract should either be preserved or shimmed during migration so the Go client can move with minimal churn

### Reusable pieces from `Sheet2Tube`

The most valuable reusable work is not the alignment algorithm alone. It is the service runtime discipline already implemented there:

- auth token handling
- health registration
- structured request errors with stable codes
- stats reporting
- inflight request limiting
- deadline-aware execution
- model prewarm hooks

That runtime should inform or be reused by the Dictator service rather than rebuilt from scratch.

## Holistic Centralization Recommendation

### What should be centralized

Centralize execution capabilities behind one shared speech service API:

- `TranscriptionService`
  - raw speech-to-text for dictation and timeline text extraction
- `AlignmentService`
  - transcript-to-audio forced alignment with word timings and SRT
- `VoiceService`
  - reference extraction
  - voice synthesis
- `ArtifactService`
  - large binary upload/download
- health and stats endpoints

### What should stay outside the central speech service

Do not centralize application-specific edge contracts:

- `llm-proxy` should keep:
  - `/dictate`
  - shared-secret query auth
  - HTTP status mapping
  - multipart compatibility behavior
- `Sheet2Tube` should keep:
  - `/reel/api/jobs`
  - SSE streams
  - job persistence
  - `client_job_id` correlation
  - app-specific artifact URLs and lifecycle

Those are adapter/orchestration concerns, not model-execution concerns.

### Recommended service surface after expansion

Suggested logical gRPC surface:

```proto
service ArtifactService {
  rpc UploadArtifact(stream UploadArtifactChunk) returns (UploadArtifactResponse);
  rpc DownloadArtifact(DownloadArtifactRequest) returns (stream DownloadArtifactChunk);
}

service TranscriptionService {
  rpc Transcribe(TranscribeRequest) returns (TranscribeResponse);
}

service AlignmentService {
  rpc AlignTranscript(stream AlignTranscriptChunk) returns (AlignTranscriptResponse);
  rpc GetStats(StatsRequest) returns (StatsResponse);
}

service VoiceService {
  rpc ExtractReferenceSample(ExtractReferenceSampleRequest)
      returns (ExtractReferenceSampleResponse);
  rpc SynthesizeSpeech(SynthesizeSpeechRequest)
      returns (SynthesizeSpeechResponse);
}
```

Recommended compatibility rule:

- keep `Sheet2Tube`'s current `svg_tools.audio_to_text.v1.AudioToText` service available through a compatibility adapter first
- migrate clients later to a neutral package namespace such as `speech.v1`

### Should this be one deployable service?

Logically, yes.

Physically, maybe not at first.

Reasons to avoid an immediate single-process monolith:

- alignment currently depends on `whisperx==3.3.0`, `torch>=2.6`, and `torchaudio`
- Dictator currently depends on `openai-whisper`, `pyannote.audio`, and `coqui-tts`
- GPU memory pressure and model warmup profiles are very different between alignment and XTTS
- dependency compatibility across these stacks is plausible on Python `3.11.8`, but not yet proven in this repository

Recommended rollout:

1. unify the codebase, protobuf strategy, runtime helpers, and service ownership
2. keep separate deploy modes or binaries for:
   - transcription/voice
   - alignment
3. use specialized workers or executors behind the shared API boundary so alignment, ASR/diarization, and TTS can be scaled independently
4. merge into one runtime only after dependency compatibility and GPU contention are validated

That gives centralization without forcing premature operational coupling.

### Updated recommendation

The answer is not "too dissimilar to centralize." The answer is:

- centralize the speech execution layer
- do not centralize the outer HTTP/job APIs
- centralize runtime patterns immediately
- centralize deployment only after dependency and capacity validation

This is the highest-leverage path with the lowest migration risk.

## Strong Encapsulation Plan

### Architectural goal

Turn Dictator from a set of top-level scripts into a package-based execution service with:

- explicit transport boundaries
- explicit domain boundaries
- explicit backend interfaces
- explicit runtime configuration
- explicit request/error contracts

The architectural model should borrow heavily from `Sheet2Tube`'s Python gRPC service runtime, especially its separation between:

- runtime/bootstrap concerns
- request validation
- transport-specific error mapping
- model execution
- metrics and health

### Target layering

Use this dependency direction:

1. `contracts`
   - protobufs, DTOs, public request/response types
2. `transport`
   - gRPC handlers, optional HTTP adapter(s), interceptors
3. `application`
   - orchestration services for transcription, alignment, extraction, synthesis
4. `domain`
   - pure audio/text logic, scoring, chunking, SRT/timeline generation
5. `infrastructure`
   - model backends, FFmpeg helpers, artifact store, temp files
6. `runtime`
   - auth, health, metrics, config, structured request errors, logging bootstrap

Rules:

- transport must not import concrete model libraries directly
- domain logic must not know about gRPC or HTTP
- infrastructure must be injected behind interfaces
- runtime config must be loaded once at startup and passed down explicitly

### Runtime patterns to borrow from `Sheet2Tube`

Adopt or port the same runtime concepts already proven in `services/mediajobs_runtime` and the alignment server:

- `ServiceRequestError`-style typed request failures
- auth helpers that accept `Authorization: Bearer` and `x-api-key`
- gRPC health registration
- stats reporting with canonical fields
- bounded inflight semaphores
- startup prewarm hooks
- explicit `ServerConfig` objects instead of scattered globals

These are better foundations for a long-running service than the current script-level logging, prompting, and exception flow.

### Encapsulation milestones

#### Milestone A: package extraction

- move all reusable logic out of `main.py` and `extract.py`
- leave the CLIs as tiny wrappers only
- remove side effects from import time

#### Milestone B: backend isolation

- wrap Whisper, WhisperX, pyannote, XTTS, and FFmpeg behind interfaces
- support mock backends in tests
- make cache/model paths configurable

#### Milestone C: runtime unification

- add runtime helpers for auth, health, errors, config, and metrics
- make all RPC handlers use the same error and deadline model

#### Milestone D: transport isolation

- add gRPC handlers only after the application layer is callable directly from tests
- add optional HTTP compatibility adapters after the gRPC contracts stabilize

### Resulting repository shape

The ideal end state is one repository that can produce multiple speech-service binaries from the same codebase:

- `speech-api` or `speech-server`
- `speech-alignment-server` if split deployment remains necessary
- CLI wrappers for local/manual workflows

That keeps ownership centralized even if deployment is temporarily split.

## Forced Alignment Capability Plan

### Objective

Add forced alignment to Dictator without regression relative to the current `Sheet2Tube` implementation.

This means the target is not merely "some alignment support." The target is feature and operational parity with the current `AudioToText` service.

### Non-negotiable parity requirements

The new alignment capability must preserve these current `Sheet2Tube` properties:

- streamed WAV input plus transcript input
- word-level output with timestamps
- SRT generation
- language support parity
- punctuation handling parity
- auth support parity
- health and stats support parity
- inflight concurrency limiting
- deadline and timeout behavior
- stable machine-readable error codes or a compatibility adapter that preserves them
- Go client compatibility during migration

### Recommended implementation strategy

Do not reimplement forced alignment from scratch inside Dictator first.

Instead:

1. extract or vendor the existing WhisperX-based alignment core from `Sheet2Tube`
2. wrap it in a neutral `AlignmentBackend`
3. expose it from the new speech service
4. preserve the old `AudioToText` gRPC contract through a compatibility surface
5. migrate callers only after parity tests pass

This reduces algorithmic regression risk and preserves the current operational behavior.

### Alignment module design

Suggested internal split:

- `alignment/models.py`
  - request/response dataclasses
  - normalized word schema
- `alignment/normalization.py`
  - transcript normalization
  - punctuation handling
  - language inference helpers
- `alignment/whisperx_backend.py`
  - model loading
  - cache management
  - execution
- `alignment/srt.py`
  - SRT generation from normalized word schema
- `alignment/service.py`
  - application-layer orchestration
  - validation
  - timeout handling

### Schema strategy

The current systems use different word schemas:

- Dictator transcription: `content`, `start`, `end`
- `Sheet2Tube` alignment: `text`, `start_seconds`, `end_seconds`

Pick one normalized internal schema for the new service, then map at the edge.

Recommended normalized internal schema:

- `text`
- `start_seconds`
- `end_seconds`

Then provide adapters:

- Dictator timeline adapter
- `Sheet2Tube` compatibility adapter
- future HTTP/JSON adapters

### Migration steps for alignment

#### Step 1: isolate `Sheet2Tube` alignment core

- move reusable alignment logic into a neutral module
- keep `Sheet2Tube` tests running against that extracted logic

#### Step 2: add compatibility tests

Build black-box parity tests that compare old and new behavior for:

- validation errors
- status codes
- auth behavior
- transcript normalization
- punctuation modes
- language defaults
- SRT output
- timeout and inflight behavior

#### Step 3: integrate into Dictator runtime

- add `AlignmentService` to the new speech server
- reuse the shared runtime helpers
- add model prewarm and cache configuration

#### Step 4: provide legacy gRPC surface

- continue serving `svg_tools.audio_to_text.v1.AudioToText`
- internally route it to the new alignment application service

#### Step 5: migrate clients

- switch `Sheet2Tube` Go clients only after parity is demonstrated
- then consider moving clients to a neutral `speech.v1.AlignmentService`

### No-degradation acceptance criteria

Forced alignment is ready only when all of the following are true:

1. `Sheet2Tube`'s current gRPC integration tests pass unchanged or via a compatibility harness.
2. The new service returns equivalent aligned words and SRT for representative fixtures.
3. The auth, health, stats, timeout, and inflight behavior match the current service contract.
4. There is no regression in supported languages or punctuation behavior.
5. The new runtime is request-safe and does not reintroduce CLI-style global process behavior.

## Migration Plan

### Current implementation status

The repo now has an initial implementation baseline, not just a plan:

- Phase 1 is complete:
  - reusable package modules exist under `dictator/`
  - `main.py`, `extract.py`, and `whisper_service.py` are compatibility wrappers
  - forced alignment has been extracted into a first-class alignment module
- Phase 2 is partially complete:
  - typed request/result models exist for the packaged services
  - a long-lived execution runtime now caches Whisper, diarization, XTTS, and alignment backends
  - one legacy compatibility edge remains in `extract.py`: the CLI still uses `signal.alarm`
- Phase 3 is partially complete:
  - a local-disk `ArtifactStore` exists for transport use
- Phase 4 is partially complete:
  - versioned protobufs exist under `proto/dictator/speech/v1/`
  - generated Python stubs exist under `dictator/speech/v1/`
- Phase 5 is partially complete:
  - an initial gRPC server scaffold exists behind `serve.py`
  - `ArtifactService`, `TranscriptionService`, `AlignmentService`, `VoiceService`, and `RuntimeService` are wired
  - health registration, auth token checks, metrics, and inflight limiting are present

What is still missing is end-to-end execution validation with the real speech stack installed, plus production hardening around deadlines, retention, and compatibility adapters.

### Phase 1: Extract a real library layer

Goal: move reusable logic out of script entrypoints without changing behavior.

Tasks:

- add a first-class transcription module instead of only `transcribe_words(...)`
- move text utilities from `main.py` into a synthesis module
- move FFmpeg normalization and concat logic into audio helpers
- move extraction orchestration from `extract.py` into an `ExtractionService`
- move Whisper loading and transcription into a backend class
- identify the reusable `Sheet2Tube` alignment core and extract it into a neutral alignment module with compatibility tests
- replace direct temp-path usage like `_tts_chunks/` with per-request temp directories
- replace interactive overwrite prompts with explicit behavior flags

Deliverable:

- the current CLIs still work, but now call library services

### Phase 2: Make execution request-safe

Goal: make the code safe to run inside a long-lived server.

Tasks:

- introduce structured request/response dataclasses or pydantic models
- inject model backends instead of instantiating them deep inside functions
- replace `signal.alarm` timeouts with cancellable request-level timeouts
- add cleanup guarantees for temp files on success and failure
- isolate logging setup at process startup only
- move torch/warnings process setup into explicit application bootstrap

Deliverable:

- pure service objects callable from tests without global side effects

### Phase 3: Add artifact storage

Goal: stop exposing local filesystem paths as the API contract.

Tasks:

- add `ArtifactStore` abstraction
- implement a local-disk store first under something like `var/artifacts/`
- assign IDs, MIME types, sizes, and metadata
- store generated WAV and optional timeline artifacts
- add retention and cleanup policy

Deliverable:

- service code talks in artifact IDs, not arbitrary paths

### Phase 4: Define and generate protobufs

Goal: formalize the network contract.

Tasks:

- create neutral protobuf packages such as `proto/speech/v1/*.proto`
- generate Python stubs as part of the build
- add versioned package names from the start
- add either a compatibility adapter or a legacy proto package for `Sheet2Tube` alignment clients
- standardize shared timeline/word schema or add explicit mappers between alignment and transcription payloads
- define clear error mapping for:
  - invalid arguments
  - missing artifacts
  - unsupported media
  - model failures
  - timeout/deadline exceeded

Deliverable:

- generated gRPC server and client stubs checked into the build flow

### Phase 5: Implement the gRPC server

Goal: expose the existing capabilities through a stable service boundary.

Tasks:

- use `grpc.aio` or a standard gRPC server with a bounded executor
- implement `ArtifactService`
- implement `TranscriptionService`
- implement `AlignmentService`
- implement `VoiceService`
- add health and readiness endpoints
- add interceptors for logging, deadline handling, and error translation
- configure max upload/download size and streaming chunk sizes
- prefer the `Sheet2Tube` alignment runtime patterns for auth, health, stats, and inflight control

Deliverable:

- a running gRPC server that can transcribe, align, extract, and synthesize

### Phase 6: Operational hardening

Goal: make the service viable beyond a developer machine.

Tasks:

- add concurrency limits per capability/model/GPU
- add startup warmup options
- add request IDs and structured logs
- emit latency metrics for:
  - decode
  - transcription
  - diarization
  - alignment
  - synthesis
  - normalization
- add disk-usage monitoring for artifacts and temp files
- make model/cache paths configurable
- ensure startup works without assuming a writable home directory

Deliverable:

- observable, controllable production behavior

## Testing Plan

### Preserve current unit coverage

Keep and adapt the current tests for:

- duration parsing
- chunk building
- word extraction
- window selection
- synthesis timeline behavior

### Add service-focused tests

- unit tests for request validation and artifact-store behavior
- gRPC contract tests against an in-process server
- integration tests with mocked model backends
- one opt-in end-to-end test suite for real FFmpeg plus real models
- compatibility tests for:
  - `llm-proxy` dictation adapter behavior
  - `Sheet2Tube` alignment client behavior

### Important regression cases

- concurrent synthesis requests do not collide in temp directories
- cancellation cleans up partial artifacts
- large uploads stream successfully
- empty or corrupt audio returns gRPC `INVALID_ARGUMENT`
- overlong jobs respect deadlines or server-side timeouts

## Risks And Design Notes

### 1. Unary-only RPCs will age poorly

They are acceptable for a narrow internal v1, but the source recordings and generated WAV files in this repository are already large enough that streaming artifact transfer is the safer default.

### 2. Timeline generation currently retranscribes the final output

This is simple and already implemented, but expensive. Keep it for parity in the first service version, then consider emitting chunk-aligned timeline data directly from synthesis plus optional alignment refinement later.

### 3. GPU memory pressure will dominate scaling

The service should treat the GPU as the constrained resource, not CPU threads. Bounded concurrency is more important than maximizing request parallelism.

### 4. Packaging needs cleanup early

The repository currently behaves like a collection of top-level scripts. A service implementation will be easier to test and deploy once the code is moved into an importable package with a clear entrypoint.

### 5. Centralizing edge contracts would be a mistake

`llm-proxy` and `Sheet2Tube` already have application-specific contracts above the model layer. Replacing those with direct speech-service ownership would increase migration cost without improving the core speech runtime.

## Recommended First Milestone

If the goal is to move quickly without rewriting everything at once, the best first milestone is:

1. extract a package-based library layer from the existing scripts
2. add a real transcription capability alongside extraction and synthesis
3. reuse the `Sheet2Tube` Python service runtime patterns for auth, health, stats, and request errors
4. add a local artifact store
5. define protobufs for transcription, alignment, extraction, synthesis, and download
6. implement compatibility adapters for:
   - `llm-proxy` `/dictate`
   - `Sheet2Tube` alignment gRPC
7. keep the CLI wrappers as compatibility shims

That gets the codebase to a usable centralized speech platform with the smallest architectural jump.

## Baseline Notes From This Analysis

- No `docs/` directory existed before this document.
- The repository is small enough that a staged refactor is realistic.
- The current shell environment did not have `pytest` installed, and `python -m unittest` failed to run the full suite because required runtime modules such as `numpy` were not installed in the active interpreter. The migration plan should therefore treat reproducible environment setup as part of the service work.
