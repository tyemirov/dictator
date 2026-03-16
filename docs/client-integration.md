# Dictator Client Integration Guide

This document describes the current client-facing contract for Dictator's gRPC API, the expected integration path, and the recommended usage patterns.

## Scope

The Python helper surface currently exported from `dictator.client` is:

- `DictationClient`
- `DiarizationClient`
- `SubtitleClient`
- `AlignmentClient`
- `SynthesisClient`
- `ReferenceSampleClient`
- `RemoteJobFailedError`

Result and job dataclasses are also exported for those clients.

The protobuf definitions and generated gRPC stubs are the authoritative contract.

The maintained Python clients are the intended ergonomic layer for Python integrations because they handle:

- artifact upload-first request shaping
- async submit/get/wait polling
- stable error surfacing
- compatibility fallback where that is part of the client contract

For non-Python integrations, generated stubs are the right baseline. Recreate the same thin wrapper pattern in that language rather than embedding job polling and upload orchestration ad hoc throughout application code.

## Service configuration and auth

For service startup, `config.yml` is the source of truth.

- `serve.py` reads only `--config`
- environment variables are used only to resolve `${...}` placeholders inside the config file
- the checked-in `config.yml` enumerates the supported service settings

Clients authenticate with gRPC metadata:

```python
metadata = (("x-dictator-token", token),)
```

Typical channel setup:

```python
import grpc

from dictator.client import DictationClient

channel = grpc.insecure_channel("127.0.0.1:50051")
client = DictationClient(channel, metadata=(("x-dictator-token", token),))
```

For direct artifact upload/download or unsupported custom flows, use the generated stubs alongside the maintained clients.

## Transport model

Dictator is artifact-first.

The intended integration path is:

1. upload input audio as an artifact
2. call a unary business RPC that references the uploaded artifact id
3. receive structured output and any derived artifact ids
4. download derived artifacts only when needed

The convenience clients perform the upload step automatically.

## Universal async job contract

All slow endpoints that support jobs follow the same shape:

- `submit_*_job(...)`
- `get_*_job(job_id)`
- `wait_for_*_job(job_id, timeout_seconds=..., poll_interval_seconds=...)`

Submit methods return a `*Job` object with:

- `job_id`
- `state`
- `source_artifact_id`

Poll methods return the same plus:

- `error_code`
- `error_message`
- `created_at_unix_seconds`
- `started_at_unix_seconds`
- `finished_at_unix_seconds`
- `result` on success

The shared polling helper treats any state ending in:

- `_SUCCEEDED` as terminal success
- `_FAILED` as terminal failure

`wait_for_*_job(...)` behavior:

- returns the terminal job on success
- raises `RemoteJobFailedError` on remote job failure
- raises `TimeoutError` if the deadline is exceeded
- accepts `timeout_seconds=None` for no deadline

## Blocking convenience methods

Each client also exposes blocking convenience methods that hide explicit job polling.

These methods:

- upload the audio artifact
- submit the async job
- wait for completion
- return the final result object

This is the default integration path for most callers.

### Compatibility fallback

The following convenience methods first try the async contract and fall back to the synchronous RPC when the server returns `UNIMPLEMENTED` or `"... job manager is not configured"`:

- `DictationClient.dictate_file(...)`
- `DictationClient.dictate_bytes(...)`
- `DiarizationClient.diarize_file(...)`
- `DiarizationClient.diarize_bytes(...)`
- `SubtitleClient.render_file(...)`
- `SubtitleClient.render_bytes(...)`
- `AlignmentClient.align_file(...)`
- `AlignmentClient.align_bytes(...)`

This fallback exists for compatibility with older or partially configured servers. New integrations should treat the async path as primary.

`SynthesisClient.synthesize(...)` and `ReferenceSampleClient.extract_file(...)` / `extract_bytes(...)` use the async job path directly and do not provide a sync fallback.

## Endpoint-specific contracts

### Dictation

Client:

- `DictationClient`

Blocking methods:

- `dictate_file(...)`
- `dictate_bytes(...)`

Async methods:

- `submit_dictate_file_job(...)`
- `submit_dictate_bytes_job(...)`
- `get_dictation_job(...)`
- `wait_for_dictation_job(...)`

Important arguments:

- `model_size`
- `language_code`
- `autodetect_language`
- `include_word_segments`

Rules:

- `language_code` and `autodetect_language=True` must not both be set
- if no `language_code` is given, set `autodetect_language=True`

Result:

- `text`
- `language_code`
- `artifact_id`
- optional `words`

### Diarization

Client:

- `DiarizationClient`

Blocking methods:

- `diarize_file(...)`
- `diarize_bytes(...)`

Async methods:

- `submit_diarize_file_job(...)`
- `submit_diarize_bytes_job(...)`
- `get_diarization_job(...)`
- `wait_for_diarization_job(...)`

Important arguments:

- `include_words`
- `include_utterances`
- `include_speakers`
- `include_speaker_segments`
- `utterance_gap_seconds`
- `persist_json_artifact`

Best practice:

- set `persist_json_artifact=True` if you need a durable diarization artifact id for later retrieval or cross-process handoff

Result:

- `text`
- `language_code`
- `source_artifact_id`
- `diarization`
- optional `diarization_artifact_id`

### Subtitles

Client:

- `SubtitleClient`

Blocking methods:

- `render_file(...)`
- `render_bytes(...)`

Async methods:

- `submit_render_file_job(...)`
- `submit_render_bytes_job(...)`
- `get_subtitle_job(...)`
- `wait_for_subtitle_job(...)`

Important arguments:

- `granularity`: `"words"` or `"sentences"`
- `group_size`
- `source_text`
- `source_text_file`
- `source_text_name`
- `include_srt_text`

Rules:

- `source_text` and `source_text_file` must not both be set
- when source text is provided, the result mode is `forced_alignment`
- otherwise the result mode is `transcription`

Result:

- `language_code`
- `mode`
- `granularity`
- `group_size`
- `source_artifact_id`
- `srt_artifact_id`
- `srt_text`
- `cues`

### Speech synthesis

Client:

- `SynthesisClient`

Blocking methods:

- `synthesize(...)`

Async methods:

- `submit_synthesize_job(...)`
- `get_synthesis_job(...)`
- `wait_for_synthesis_job(...)`

Important arguments:

- `speaker_artifact_id`
- `text`
- `text_artifact_id`
- `language_code`
- `max_duration_seconds`
- `include_timeline`
- `synthesis_engine`
- `speaker_transcript_text`

Rules:

- `speaker_artifact_id` is required
- set exactly one of `text` or `text_artifact_id`
- `speaker_transcript_text` should be provided when known for higher quality voice conditioning

Result:

- `audio_artifact_id`
- `audio_duration_seconds`
- optional `timeline_artifact_id`
- `chunk_count`

Minimal Python example:

```python
import grpc

from dictator.client import SynthesisClient
from dictator.speech.v1 import artifacts_pb2, artifacts_pb2_grpc


def iter_upload(filename: str, media_type: str, payload: bytes):
    yield artifacts_pb2.UploadArtifactChunk(
        metadata=artifacts_pb2.UploadArtifactMetadata(
            filename=filename,
            media_type=media_type,
        )
    )
    yield artifacts_pb2.UploadArtifactChunk(content=payload)


token = "your-token"
metadata = (("x-dictator-token", token),)

channel = grpc.insecure_channel("127.0.0.1:50051")
artifact_stub = artifacts_pb2_grpc.ArtifactServiceStub(channel)
upload = artifact_stub.UploadArtifact(
    iter_upload("speaker.wav", "audio/wav", open("speaker.wav", "rb").read()),
    metadata=metadata,
)

client = SynthesisClient(channel, metadata=metadata)
result = client.synthesize(
    speaker_artifact_id=upload.artifact.artifact_id,
    text="Hello from Dictator.",
    language_code="en",
    speaker_transcript_text="The transcript of the reference sample.",
)
print(result.audio_artifact_id)
```

### Alignment

Client:

- `AlignmentClient`

Blocking methods:

- `align_file(...)`
- `align_bytes(...)`

Async methods:

- `submit_align_file_job(...)`
- `submit_align_bytes_job(...)`
- `get_alignment_job(...)`
- `wait_for_alignment_job(...)`

Transcript inputs:

- `transcript_text`
- `transcript_file`
- `transcript_artifact_id`

Rule:

- exactly one transcript source must be set

Other important arguments:

- `language_code`
- `remove_punctuation`
- `include_srt_text`

Result:

- `language_code`
- `source_artifact_id`
- `srt_artifact_id`
- `srt_text`
- `words`

### Reference sample extraction

Client:

- `ReferenceSampleClient`

Blocking methods:

- `extract_file(...)`
- `extract_bytes(...)`

Async methods:

- `submit_extract_file_job(...)`
- `submit_extract_bytes_job(...)`
- `get_reference_sample_job(...)`
- `wait_for_reference_sample_job(...)`

Important arguments:

- `model_size`
- `language_code`
- `duration_seconds`
- `max_speech_rate`
- `min_centroid_hz`
- `max_centroid_hz`

Result:

- `sample_artifact_id`
- `trim_start_seconds`
- `trim_end_seconds`
- `window_start_seconds`
- `window_end_seconds`
- `dominant_speaker_word_count`

## Recommended integration path

Use the blocking convenience methods unless you have a concrete reason not to.

Good default pattern:

```python
import grpc

from dictator.client import AlignmentClient

channel = grpc.insecure_channel("127.0.0.1:50051")
client = AlignmentClient(channel, metadata=(("x-dictator-token", token),))
result = client.align_file(
    audio_path,
    transcript_file=transcript_path,
    language_code="en",
)
print(result.srt_artifact_id)
print(result.srt_text)
```

Use explicit jobs when:

- the caller owns its own polling loop
- the work is long-running and should survive process boundaries
- the caller wants to persist `job_id` and resume later
- the caller wants separate submit and observe phases

Explicit async pattern:

```python
job = client.submit_render_file_job(audio_path, autodetect_language=True)
job = client.wait_for_subtitle_job(job.job_id, timeout_seconds=900.0)
result = job.result
```

## Best practices

- Prefer convenience methods for normal request/response application code.
- Prefer explicit jobs for workflow engines, queue consumers, or external orchestrators.
- Persist `job_id` if completion may outlive the current process.
- Persist artifact ids when downstream steps need to fetch generated SRT, diarization JSON, or extracted samples later.
- Use `timeout_seconds=None` only when an external supervisor already owns cancellation.
- Keep `poll_interval_seconds` coarse enough to avoid unnecessary load; `1.0` second is the current default.
- Treat the sync fallback path as compatibility only. Do not depend on it for new deployments.
- When using subtitles with known source text, provide the source text so the service performs forced alignment instead of plain transcription.
- For alignment, validate transcript-source selection on the caller side as well: exactly one of inline text, transcript file, or transcript artifact id.

## gRPC-backed CLI wrappers

The following CLIs now use the client layer:

- `dictate.py`
- `subtitle.py`
- `align.py`

Their wait behavior is driven by `config.yml`:

- `grpc.job_wait_timeout_seconds`
- `grpc.job_poll_interval_seconds`

This keeps timeout and poll policy in service configuration rather than as per-invocation CLI flags.

## Current truth sources

If this document and the code diverge, trust the code and update this document immediately.

Primary truth sources:

- `dictator/client/*.py`
- `proto/dictator/speech/v1/*.proto`
- transport integration tests in `tests/test_grpc_transport_integration.py`
- client tests in `tests/test_client_job_helpers.py` and `tests/test_client_alignment.py`
