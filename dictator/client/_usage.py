"""Validate native measurements at the response boundary."""

from google.protobuf.message import Message

from dictator.audio.usage import InputAudioUsage


def input_audio_usage_from_response(response: Message) -> InputAudioUsage:
    if not response.HasField("input_audio_usage"):
        raise ValueError("successful response requires input_audio_usage")
    return InputAudioUsage(
        sample_count=response.input_audio_usage.sample_count,
        sample_rate_hz=response.input_audio_usage.sample_rate_hz,
    )
