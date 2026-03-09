"""Small HTTP bridge for a browser-based Dictator voice cloning demo."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
from typing import Callable
from urllib.parse import urlparse

import grpc

from dictator.audio.ffmpeg_ops import audio_to_wav
from dictator.speech.v1 import artifacts_pb2, artifacts_pb2_grpc, voice_pb2, voice_pb2_grpc

INDEX_HTML_PATH = Path(__file__).with_name("index.html")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
UPLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_LANGUAGE_CODE = "en"
DEFAULT_OUTPUT_FILENAME = "reading-in-your-voice.wav"
DICTATOR_URL_ENV = "VOICE_CLONE_DICTATOR_URL"
DICTATOR_AUTH_TOKEN_ENV = "DICTATOR_GRPC_AUTH_TOKEN"
DEFAULT_SYNTHESIS_ENGINE_ID = "xtts"

VOICE_SAMPLE_TEXT = (
    "I grew up near a busy street, so even now I sleep best with a little noise in the distance. "
    "My friends know I speak quickly when I am excited, slow down when I am serious, and laugh "
    "before I finish the punch line. On cold mornings I want strong coffee, warm light, and ten "
    "quiet minutes to think. If you know me well, you can hear the difference between my polite "
    "voice, my tired voice, and the one I use when I am truly delighted."
)

GENESIS_EXCERPT = """Genesis 1:1-10, King James Version.
In the beginning God created the heaven and the earth.
And the earth was without form, and void; and darkness was upon the face of the deep.
And the Spirit of God moved upon the face of the waters.
And God said, Let there be light: and there was light.
And God saw the light, that it was good: and God divided the light from the darkness.
And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day.
And God said, Let there be a firmament in the midst of the waters, and let it divide the waters from the waters.
And God made the firmament, and divided the waters which were under the firmament from the waters which were above the firmament: and it was so.
And God called the firmament Heaven. And the evening and the morning were the second day.
And God said, Let the waters under the heaven be gathered together unto one place, and let the dry land appear: and it was so.
And God called the dry land Earth; and the gathering together of the waters called he Seas: and God saw that it was good.
"""

ALICE_EXCERPT = """Alice's Adventures in Wonderland, Chapter I.
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do.
Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it.
"And what is the use of a book," thought Alice, "without pictures or conversations?"
So she was considering in her own mind, as well as she could, whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies."""

WOODS_EXCERPT = """Stopping by Woods on a Snowy Evening, by Robert Frost.
Whose woods these are I think I know.
His house is in the village though;
He will not see me stopping here
To watch his woods fill up with snow.

My little horse must think it queer
To stop without a farmhouse near
Between the woods and frozen lake
The darkest evening of the year.

He gives his harness bells a shake
To ask if there is some mistake.
The only other sound's the sweep
Of easy wind and downy flake.

The woods are lovely, dark and deep,
But I have promises to keep,
And miles to go before I sleep,
And miles to go before I sleep."""


class ExampleRequestError(ValueError):
    """Raised when the example request cannot be fulfilled."""


@dataclass(frozen=True)
class GrpcTarget:
    authority: str
    secure: bool


@dataclass(frozen=True)
class VoiceCloneResult:
    filename: str
    media_type: str
    content: bytes


@dataclass(frozen=True)
class RenderPreset:
    preset_id: str
    label: str
    filename: str
    text: str


@dataclass(frozen=True)
class SynthesisEngineOption:
    engine_id: str
    label: str
    proto_value: int
    speaker_transcript_text: str | None = None


DEFAULT_RENDER_PRESET_ID = "genesis"
RENDER_PRESETS = {
    "genesis": RenderPreset(
        preset_id="genesis",
        label="Genesis",
        filename="genesis-in-your-voice.wav",
        text=GENESIS_EXCERPT,
    ),
    "alice": RenderPreset(
        preset_id="alice",
        label="Alice in Wonderland",
        filename="alice-in-your-voice.wav",
        text=ALICE_EXCERPT,
    ),
    "woods": RenderPreset(
        preset_id="woods",
        label="Dark Woods",
        filename="dark-woods-in-your-voice.wav",
        text=WOODS_EXCERPT,
    ),
}

SYNTHESIS_ENGINES = {
    "xtts": SynthesisEngineOption(
        engine_id="xtts",
        label="XTTS",
        proto_value=voice_pb2.SYNTHESIS_ENGINE_XTTS,
    ),
    "qwen3": SynthesisEngineOption(
        engine_id="qwen3",
        label="Qwen3-TTS",
        proto_value=voice_pb2.SYNTHESIS_ENGINE_QWEN3,
        speaker_transcript_text=VOICE_SAMPLE_TEXT,
    ),
}


def load_index_html() -> str:
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


def parse_grpc_target(raw_target: str) -> GrpcTarget:
    target = (raw_target or "").strip()
    if not target:
        raise ExampleRequestError("Dictator URL is required.")
    if "://" not in target:
        if any(character in target for character in "/?#"):
            raise ExampleRequestError("Dictator URL must not include a path, query, or fragment.")
        return GrpcTarget(authority=target, secure=False)
    parsed = urlparse(target)
    if parsed.scheme not in {"grpc", "grpcs", "http", "https"}:
        raise ExampleRequestError("Dictator URL must use grpc://, grpcs://, http://, https://, or host:port.")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ExampleRequestError("Dictator URL must point to the Dictator host and port only.")
    if not parsed.netloc:
        raise ExampleRequestError("Dictator URL must include a host.")
    return GrpcTarget(authority=parsed.netloc, secure=parsed.scheme in {"grpcs", "https"})


def resolve_bridge_target(configured_target: str | None) -> str:
    target = (configured_target or "").strip()
    if target:
        return target
    raise ExampleRequestError("Dictator bridge target is not configured.")


def resolve_bridge_auth_token(configured_token: str | None) -> str:
    token = (configured_token or "").strip()
    if token:
        return token
    raise ExampleRequestError("Dictator bridge auth token is not configured.")


def resolve_render_preset(raw_preset_id: str | None) -> RenderPreset:
    preset_id = (raw_preset_id or DEFAULT_RENDER_PRESET_ID).strip() or DEFAULT_RENDER_PRESET_ID
    preset = RENDER_PRESETS.get(preset_id)
    if preset is None:
        raise ExampleRequestError(f"Unknown reading selection: {preset_id}")
    return preset


def resolve_synthesis_engine(raw_engine_id: str | None) -> SynthesisEngineOption:
    engine_id = (raw_engine_id or DEFAULT_SYNTHESIS_ENGINE_ID).strip().lower() or DEFAULT_SYNTHESIS_ENGINE_ID
    engine = SYNTHESIS_ENGINES.get(engine_id)
    if engine is None:
        raise ExampleRequestError(f"Unknown synthesis engine: {engine_id}")
    return engine


def build_auth_metadata(auth_token: str) -> list[tuple[str, str]]:
    token = (auth_token or "").strip()
    if not token:
        raise ExampleRequestError("Auth token is required.")
    return [("authorization", f"Bearer {token}")]


def create_channel(target: GrpcTarget) -> grpc.Channel:
    if target.secure:
        return grpc.secure_channel(target.authority, grpc.ssl_channel_credentials())
    return grpc.insecure_channel(target.authority)


def iter_upload_chunks(filename: str, media_type: str, payload: bytes):
    yield artifacts_pb2.UploadArtifactChunk(
        metadata=artifacts_pb2.UploadArtifactMetadata(
            filename=filename,
            media_type=media_type,
        )
    )
    for start in range(0, len(payload), UPLOAD_CHUNK_BYTES):
        yield artifacts_pb2.UploadArtifactChunk(content=payload[start : start + UPLOAD_CHUNK_BYTES])


def upload_artifact(
    stub: artifacts_pb2_grpc.ArtifactServiceStub,
    *,
    filename: str,
    media_type: str,
    payload: bytes,
    metadata: list[tuple[str, str]],
) -> str:
    response = stub.UploadArtifact(iter_upload_chunks(filename, media_type, payload), metadata=metadata)
    return response.artifact.artifact_id


def download_artifact(
    stub: artifacts_pb2_grpc.ArtifactServiceStub,
    *,
    artifact_id: str,
    metadata: list[tuple[str, str]],
) -> VoiceCloneResult:
    chunks = stub.DownloadArtifact(
        artifacts_pb2.DownloadArtifactRequest(
            artifact_id=artifact_id,
            chunk_size=DOWNLOAD_CHUNK_BYTES,
        ),
        metadata=metadata,
    )
    filename = DEFAULT_OUTPUT_FILENAME
    media_type = "application/octet-stream"
    content = bytearray()
    for chunk in chunks:
        if chunk.artifact.filename:
            filename = chunk.artifact.filename
        if chunk.artifact.media_type:
            media_type = chunk.artifact.media_type
        if chunk.content:
            content.extend(chunk.content)
    return VoiceCloneResult(filename=filename, media_type=media_type, content=bytes(content))


def synthesize_selected_reading(
    *,
    dictator_url: str,
    auth_token: str,
    audio_payload: bytes,
    audio_filename: str,
    audio_media_type: str,
    render_preset_id: str = DEFAULT_RENDER_PRESET_ID,
    synthesis_engine_id: str = DEFAULT_SYNTHESIS_ENGINE_ID,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    channel_factory: Callable[[GrpcTarget], grpc.Channel] = create_channel,
) -> VoiceCloneResult:
    if not audio_payload:
        raise ExampleRequestError("A recorded voice sample is required.")
    render_preset = resolve_render_preset(render_preset_id)
    synthesis_engine = resolve_synthesis_engine(synthesis_engine_id)
    target = parse_grpc_target(dictator_url)
    metadata = build_auth_metadata(auth_token)
    speaker_payload, speaker_filename, speaker_media_type = normalise_recorded_audio(
        audio_payload,
        audio_filename,
        audio_media_type,
    )
    channel = channel_factory(target)
    try:
        artifact_stub = artifacts_pb2_grpc.ArtifactServiceStub(channel)
        voice_stub = voice_pb2_grpc.VoiceServiceStub(channel)
        source_artifact_id = upload_artifact(
            artifact_stub,
            filename=speaker_filename,
            media_type=speaker_media_type,
            payload=speaker_payload,
            metadata=metadata,
        )
        synthesis_request_kwargs = {
            "speaker_artifact_id": source_artifact_id,
            "text": render_preset.text,
            "language_code": language_code,
            "synthesis_engine": synthesis_engine.proto_value,
        }
        if synthesis_engine.speaker_transcript_text is not None:
            synthesis_request_kwargs["speaker_transcript_text"] = synthesis_engine.speaker_transcript_text
        synthesis_response = voice_stub.SynthesizeSpeech(
            voice_pb2.SynthesizeSpeechRequest(**synthesis_request_kwargs),
            metadata=metadata,
        )
        result = download_artifact(
            artifact_stub,
            artifact_id=synthesis_response.audio_artifact.artifact_id,
            metadata=metadata,
        )
        return VoiceCloneResult(
            filename=render_preset.filename,
            media_type=result.media_type,
            content=result.content,
        )
    finally:
        channel.close()


synthesize_genesis_reading = synthesize_selected_reading


def decode_request_payload(raw_body: bytes) -> dict[str, str]:
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExampleRequestError("Request body must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ExampleRequestError("Request body must be a JSON object.")
    return payload


def decode_audio_base64(value: str) -> bytes:
    encoded = (value or "").strip()
    if not encoded:
        raise ExampleRequestError("Recorded audio is required.")
    prefix, separator, suffix = encoded.partition(",")
    candidate = suffix if separator and prefix.startswith("data:") else encoded
    try:
        return base64.b64decode(candidate, validate=True)
    except ValueError as exc:
        raise ExampleRequestError("Recorded audio must be base64 encoded.") from exc


def normalise_recorded_audio(
    audio_payload: bytes,
    audio_filename: str,
    audio_media_type: str,
) -> tuple[bytes, str, str]:
    if not audio_payload:
        raise ExampleRequestError("A recorded voice sample is required.")

    source_name = Path((audio_filename or "").strip() or "voice-sample.webm").name or "voice-sample.webm"
    source_stem = Path(source_name).stem or "voice-sample"
    with tempfile.TemporaryDirectory(prefix="voice_clone_demo_") as tmpdir:
        source_path = Path(tmpdir) / source_name
        output_path = Path(tmpdir) / f"{source_stem}.wav"
        source_path.write_bytes(audio_payload)
        try:
            audio_to_wav(source_path, output_path)
        except Exception as exc:  # pragma: no cover - integration safety
            raise ExampleRequestError(
                f"Recorded audio could not be converted to WAV from {audio_media_type or 'the provided media type'}."
            ) from exc
        return output_path.read_bytes(), output_path.name, "audio/wav"


def choose_download_filename(filename: str) -> str:
    cleaned = Path((filename or "").strip() or DEFAULT_OUTPUT_FILENAME).name
    return cleaned or DEFAULT_OUTPUT_FILENAME


def build_handler(
    *,
    index_html: str | None = None,
    synthesizer: Callable[..., VoiceCloneResult] = synthesize_selected_reading,
    default_dictator_url: str | None = None,
    default_auth_token: str | None = None,
):
    html = index_html if index_html is not None else load_index_html()
    configured_default_dictator_url = (
        default_dictator_url if default_dictator_url is not None else os.getenv(DICTATOR_URL_ENV, "")
    ).strip() or None
    configured_auth_token = (
        default_auth_token if default_auth_token is not None else os.getenv(DICTATOR_AUTH_TOKEN_ENV, "")
    ).strip() or None

    class VoiceCloneDemoHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: HTTPStatus, payload: dict[str, str]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, status: HTTPStatus, *, body: bytes, media_type: str, filename: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Disposition", f'attachment; filename="{choose_download_filename(filename)}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/clone":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Use application/json."})
                return
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            request_payload = self.rfile.read(content_length)
            try:
                payload = decode_request_payload(request_payload)
                dictator_url = resolve_bridge_target(configured_default_dictator_url)
                auth_token = resolve_bridge_auth_token(configured_auth_token)
                result = synthesizer(
                    dictator_url=dictator_url,
                    auth_token=auth_token,
                    audio_payload=decode_audio_base64(str(payload.get("audioBase64", ""))),
                    audio_filename=str(payload.get("audioFilename", "voice-sample.webm")),
                    audio_media_type=str(payload.get("audioMediaType", "audio/webm")),
                    render_preset_id=str(payload.get("renderPreset", DEFAULT_RENDER_PRESET_ID)),
                    synthesis_engine_id=str(payload.get("synthesisEngine", DEFAULT_SYNTHESIS_ENGINE_ID)),
                    language_code=str(payload.get("languageCode", DEFAULT_LANGUAGE_CODE))
                    or DEFAULT_LANGUAGE_CODE,
                )
            except ExampleRequestError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except grpc.RpcError as exc:
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {
                        "error": f"Dictator request failed with {exc.code().name}: {exc.details()}",
                    },
                )
                return
            self._send_bytes(
                HTTPStatus.OK,
                body=result.content,
                media_type=result.media_type,
                filename=result.filename,
            )

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return VoiceCloneDemoHandler


def serve(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), build_handler())
    print(f"Voice clone demo listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    serve(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
