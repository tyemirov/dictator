# CHANGELOG

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
