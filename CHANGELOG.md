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

## [v1.10.2] - 2026-05-21

### Features ✨
- Add explicit synthesis audio format contract for speech synthesis requests and responses.
- Expose detailed audio metadata on artifact references including container, codec, sample rate, channel count, and bit depth.
- Preserve and return upload stream status details in the Go SDK for improved error handling.

### Improvements ⚙️
- Update deployment script for improved reliability.
- Enhanced Go SDK with helper functions to properly handle streaming upload statuses.

### Bug Fixes 🐛
- Handle unknown audio format enums gracefully in client code.
- Tolerate nonnumeric ffprobe metadata to prevent failures during media probing.

### Testing 🧪
- Improve test coverage for artifact store, client job helpers, gRPC services, runtime storage, and transport layers.
- Add unit tests for Go SDK upload helper methods to ensure proper status preservation.

### Docs 📚
- Update Go SDK README with new usage notes on streaming uploads.
- Add client integration documentation describing audio format contract and metadata exposure.

## [v1.10.1] - 2026-05-07

### Features ✨
- Added new release and deploy scripts with Makefile integration for streamlined operations.
- Introduced `make release` command to cut and verify SemVer releases from the master branch.
- Introduced `make deploy` command to verify and deploy the backend through mprlab-gateway.

### Improvements ⚙️
- Updated Makefile to include release and deploy targets with customizable arguments.
- Enhanced README with detailed explanations of the new release flow and Makefile contract.
- Integrated image digest verification in deploy script to ensure consistency between release tags and latest images.

### Bug Fixes 🐛
- _No changes._

### Testing 🧪
- `make ci` is now run automatically before deployment to ensure code quality.

### Docs 📚
- Expanded the README to cover new release and deployment processes using Makefile commands.
- Added usage and detailed information for the new release and deploy scripts.

## [v1.10.0] - 2026-05-02

### Features ✨
- Added handwritten Go `Client` helper for the Dictator Speech API to simplify artifact-first speech interactions.
- Introduced `TranscribeAudio` method that uploads audio and transcribes in a single call.

### Improvements ⚙️
- Default and customizable authentication metadata keys and upload chunk sizes for client configuration.
- Streamlined upload process with chunked content sending and metadata handling in the Go client.

### Bug Fixes 🐛
- _No changes._

### Testing 🧪
- Comprehensive unit tests covering artifact upload, transcription, authentication handling, and error scenarios for the new Go client helper.

### Docs 📚
- Updated Go client README with usage details of the new handwritten `Client` helper.

## [v1.9.4] - 2026-04-29

### Features ✨
- Added opt-in full synthesis probe for blackbox testing inside the Docker image.
- Introduced environment flag to enable full Qwen3 synthesis probe requiring GPU/RAM.

### Improvements ⚙️
- Enhanced Docker entrypoint health checking with improved logging and error reporting.
- Updated blackbox probe script to support optional full synthesis roundtrip.
- Modified test Docker script to include `--full-synthesis` option and pass related environment variable.

### Bug Fixes 🐛
- _No changes._

### Testing 🧪
- Added coverage for full synthesis testing via opt-in in Docker blackbox probe.

### Docs 📚
- Updated test-docker-image.sh usage information to document `--full-synthesis` option and probe behavior.

## [v1.9.3] - 2026-04-29

### Features ✨
- Add `--platform` option to test Docker images for multi-architecture support.

### Improvements ⚙️
- Enhance Docker build and run commands to support platform specification.
- Update smoke test scripts to run on specified platforms during deployment.

### Bug Fixes 🐛
- _No changes._

### Testing 🧪
- Test the Docker image blackbox probe across multiple platforms via new option.

### Docs 📚
- _No changes._

## [v1.9.2] - 2026-04-29

### Features ✨
- Refactor GPU image publishing to be performed locally for better Buildx cache reuse.
- Publish script now requires master branch, no open PRs, and HEAD with latest SemVer tag.

### Improvements ⚙️
- Enhanced verification in publish script to validate branch, PR state, and tag consistency.
- Simplified Makefile target `publish` replaces `publish-gpu-image` with improved defaults.
- Updated documentation to clarify new publishing workflow and requirements.

### Bug Fixes 🐛
- _No changes._

### Testing 🧪
- _No changes._

### Docs 📚
- Updated README to reflect new publishing process and usage examples with `make publish`.

## [v1.9.1] - 2026-04-29

### Features ✨
- _No changes._

### Improvements ⚙️
- Stabilize CI with fixed dependency versions and cache improvements.
- Add explicit installation of CPU-specific PyTorch packages in CI.
- Upgrade protobuf tool dependencies to specific versions for consistency.

### Bug Fixes 🐛
- _No changes._

### Testing 🧪
- Disable pip caching in CI to ensure fresh installs.
- Refine test workflow setup to improve stability and reliability.

### Docs 📚
- _No changes._

## [v1.9.0] - 2026-04-29

### Features ✨
- Added support for job cancellation across all relevant job types.
- Introduced job cancellation methods in client APIs for alignment, diarization, dictation, subtitles, synthesis, and reference sample jobs.
- Enhanced gRPC service definitions and SDKs to support job cancellation operations.

### Improvements ⚙️
- Improved job management with better tracking and cleanup of job futures in the local job manager.
- Updated Makefile to improve proto generation sync check logic.
- Cleaned up demo voice clone web audio recording lifecycle and UI handling.

### Bug Fixes 🐛
- Fixed audio recording playback issues in the demo web client related to audio URL management and media stream cleanup.

### Testing 🧪
- Added new tests for asynchronous job cancellation and client job helper coverage.
- Enhanced runtime storage coverage tests for job cancellation and updates.

### Docs 📚
- Updated client integration documentation to reflect job cancellation support.

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
