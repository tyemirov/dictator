# CHANGELOG

## [Unreleased]

### Features ✨
- _No changes._

### Improvements ⚙️
- _No changes._

### Bug Fixes 🐛
- _No changes._

### Testing 🧪
- _No changes._

### Docs 📚
- _No changes._

## [v1.8.0] - 2026-03-23

### Features ✨
- Publish generated Go gRPC bindings for the authoritative `dictator.speech.v1` contract in `sdk/go/dictatorspeechv1`.

### Improvements ⚙️
- Move protobuf `go_package` ownership from a MediaOps-internal path to the Dictator-owned Go contract module.
- Extend `make proto` and CI to regenerate and verify both Python and Go gRPC artifacts from the checked-in proto sources.
- Pin `protoc` for deterministic Go stub generation across local and CI environments.

### Bug Fixes 🐛
- Remove the short-lived custom `GetReadiness` startup RPC before release and repair the post-merge proto generation / CI bootstrap regressions.

### Testing 🧪
- Add owner-side proto generation checks for the checked-in Python and Go contract artifacts.

### Docs 📚
- Document the Dictator-owned Go contract module, vendoring guidance, and pinned codegen bootstrap behavior in the README and client integration guide.

## [v1.7.0] - 2026-03-17

### Features ✨
- Adopt semantic Dictator config DSL with improved structure and strict validation
- Enhance config DSL parser with inline comment stripping
- Support inquiries bypassing inflight limiter for gRPC services

### Improvements ⚙️
- Optimize concurrency handling for powerful servers with increased default max workers and inflight limits
- Improve runtime resilience with retries on transient job file read errors
- Add safe artifact cleanup on synthesis failures
- Expand gRPC config DSL and validate unknown keys
- Restructure and rename `config.yml` keys for clarity and consistency

### Bug Fixes 🐛
- Ensure job data reading retries on race conditions to reduce read errors
- Fix gRPC services to correctly mark inquiry requests and avoid inflight limiter blocking

### Testing 🧪
- Achieve 100% coverage for resilience and concurrency-related changes
- Add extra coverage tests for gRPC config DSL and runtime resilience

### Docs 📚
- _No changes._

## [v1.6.0] - 2026-03-15

### Features ✨
- Added voice clone demo jobs with asynchronous job submission, progress tracking, and audio retrieval.
- Introduced a new voice clone web demo API supporting voice synthesis job management.
- Implemented a GPU-only container image with model prefetch and improved runtime toolchain.

### Improvements ⚙️
- Updated documentation to clarify NVIDIA GPU requirement and simplify GPU image usage.
- Improved demo voice clone web app for better job state handling and progress reporting.
- Enhanced container images with pre-baked Qwen3 voice cloning model and FlashAttention integration.

### Bug Fixes 🐛
- Fixed text cleanup in synthesis and corrected capped progress totals for job progress.
- Addressed audio artifact download issue in voice clone demo job flow.

### Testing 🧪
- Added extensive tests for client job helpers and voice clone web example.
- Increased runtime and service logic test coverage for synthesis and transport layers.

### Docs 📚
- Revised README and client integration docs to remove CPU fallback and emphasize GPU support.
- Updated demo usage instructions, environment setup, and deployment with Docker Compose.

## [v1.5.0] - 2026-03-15

### Features ✨
- Add universal async job APIs and comprehensive client support for alignment, dictation, subtitles, diarization, and voice reference samples.
- Introduce Python client `AlignmentClient` with async and sync job support plus poll-based waiting helper utilities.
- Provide detailed client integration documentation and examples for gRPC service usage and async job workflows.

### Improvements ⚙️
- Enhance async job coverage and fix regressions in async client job management.
- Add new configuration parameters for synthesis, alignment, transcription, diarization, subtitle, and reference extraction job workers and queues.
- Update CLI tools (`align.py`, `dictate.py`) to use new async job client APIs with timeout and polling controls.

### Bug Fixes 🐛
- Fix async client job regressions and CI test discovery import errors.
- Address async job coverage gaps to improve stability and reliability.

### Testing 🧪
- Add extensive tests for async client regressions, job coverage, async job endpoints, and runtime storage.
- Expand client alignment, CLI job polling config, and gRPC transport integration tests to cover new async client functionality.

### Docs 📚
- Document the client integration contract and usage patterns in `docs/client-integration.md`.
- Update README with Python client quick start and async job best practices.

## [v1.4.0]

### Features ✨
- Add asynchronous synthesis job APIs for background processing.
- Implement local persistent store for synthesis job statuses.
- Include gRPC methods to submit and query synthesis speech jobs by ID.

### Improvements ⚙️
- Guard synthesis job pending counter against start failures to maintain accurate state.
- Enhance synthesis job handling with detailed job state tracking and error reporting.

### Bug Fixes 🐛
- Reject synthesis job IDs that are invalid or contain unsupported characters.

### Testing 🧪
- Add unit tests for gRPC services and runtime storage coverage related to synthesis jobs and job status persistence.

### Docs 📚
- _No changes._

## [v1.3.0]

### Features ✨
- Refactor voice cloning system to use Qwen3-TTS exclusively.
- Browser voice clone demo updated to use Qwen3-TTS with full sample plus transcript.
- Default voice cloning model updated to `Qwen/Qwen3-TTS-12Hz-1.7B-Base` for higher quality.

### Improvements ⚙️
- GPU Docker image simplified: removed CosyVoice 3 and related dependencies; now installs `sox` and official `flash-attn` wheel for improved acceleration.
- Prefetches Qwen3 default voice cloning model during GPU image build for faster startup.
- Updated CLI to require `--sample-text` for voice cloning and default to Qwen3 engine.
- Simplified gRPC target parsing and channel creation in the web demo backend.
- Improved Dockerfiles to reduce unused packages and environment variables.
- Demo backend converts browser recordings to WAV and sends fixed transcripts to Qwen3 model.

### Bug Fixes 🐛
- Fixed Docker GPU image build by removing obsolete CosyVoice runtime install steps.
- Corrected Dockerfile package installation to include `sox` utility for audio processing.

### Testing 🧪
- _No changes._

### Docs 📚
- Updated README to reflect removal of CosyVoice 3, switch to Qwen3-TTS, and new default model.
- Refined browser demo README to explain updated voice cloning approach with Qwen3.
- Cleaned up CLI help text to remove obsolete engine options and clarify required parameters.

## [v1.2.0]

### Features ✨
- Add CosyVoice 3 synthesis engine with official zero-shot API support.
- Bake synthesis models for XTTS-v2, Qwen3-TTS, and CosyVoice 3 directly into the GPU Docker image.
- Extend CLI and web demo to support CosyVoice 3 synthesis engine selection.

### Improvements ⚙️
- Improve Qwen3 synthesis runtime performance and pack synthesis chunks by token budget.
- Require and integrate flash attention as a built-in requirement for Qwen fast attention.
- Prefetch XTTS-v2, Qwen3-TTS, and CosyVoice 3 model assets during GPU image build for faster startup.
- Refactor synthesis service runtime to cache and manage all backend engines, including CosyVoice 3.
- Update GPU Dockerfile to install and patch CosyVoice 3 runtime dependencies and submodules.

### Bug Fixes 🐛
- Fix regressions in Qwen and XTTS loader components.

### Testing 🧪
- Add new and update existing tests for synthesis service logic and CLI entry points to cover recent changes.

### Docs 📚
- Update README to document CosyVoice 3 engine support in CLI, Docker, and demos.
- Enhance demo voice clone web README and UI to include CosyVoice 3 engine option.
- Clarify Dockerfile.gpu usage notes with CosyVoice 3 integration details.
