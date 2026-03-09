# CHANGELOG

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
