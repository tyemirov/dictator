"""Map native input quantities to the canonical protobuf message."""

from dictator.audio.usage import InputAudioUsage
from dictator.speech.v1 import common_pb2


def input_audio_usage_message(usage: InputAudioUsage | None) -> common_pb2.InputAudioUsage | None:
    if usage is None:
        return None
    return common_pb2.InputAudioUsage(**usage.to_json_dict())
