# ISSUES

## Open

- [x] [API-001] Synthesis output format is implicit and not negotiable.
  - Source: MediaOps Dictator provider integration review on 2026-05-21.
  - Finding: `VoiceService.SynthesizeSpeech` returns an `audio_artifact` and duration, but `SynthesizeSpeechRequest` has no field for requested container, codec, sample rate, channel count, or bit depth. Consumers can only infer the concrete output from current runtime behavior after downloading or probing the artifact.
  - Impact: Downstream capability catalogs cannot honestly advertise multiple Dictator output formats. MediaOps had to restrict Dictator speech synthesis to the observed default provider output instead of exposing `wav_44100` or `wav_48000`.
  - Expected: Add an explicit synthesis output-audio contract, such as a `SynthesisAudioFormat` request field plus resolved output metadata in `SynthesizeSpeechResponse` and `GetSynthesizeSpeechJobResponse`.

- [x] [API-002] Artifact references do not expose audio properties needed by downstream tools.
  - Source: MediaOps Dictator provider integration review on 2026-05-21.
  - Finding: `ArtifactRef` exposes only generic artifact fields (`artifact_id`, `filename`, `media_type`, `size_bytes`, `sha256`). It does not expose audio-specific metadata such as container, codec, sample rate, channel count, bit depth, or duration.
  - Impact: Consumers that need deterministic media handling must download and inspect artifacts out of band, or hard-code assumptions about Dictator outputs. This makes provider capability reporting and post-processing brittle.
  - Expected: Add optional media metadata, either directly on artifact responses or as an audio-specific metadata message associated with artifacts returned by speech APIs.

- [ ] [SDK-001] Go SDK should preserve streaming upload status details behind a helper.
  - Source: MediaOps Dictator provider integration review on 2026-05-21.
  - Finding: Raw `ArtifactService.UploadArtifact` client streams can return `io.EOF` from `Send` before the final gRPC status is observed. Consumers must call `CloseAndRecv` to recover the server-side validation status and message.
  - Impact: Consumers that implement the stream directly can accidentally surface opaque `EOF` errors instead of actionable Dictator validation failures.
  - Expected: Provide or document a first-class SDK upload helper that sends metadata/content, handles `io.EOF` correctly, and returns the most specific upstream gRPC status.
