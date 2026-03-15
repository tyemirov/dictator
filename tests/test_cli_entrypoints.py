import importlib
import json
import sys
import tempfile
import types
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch


class CliEntrypointsTests(unittest.TestCase):
    def test_align_main_returns_early_when_user_declines_overwrite(self):
        import align

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "input.wav"
            text_path = root / "transcript.txt"
            output_path = root / "out.srt"
            input_path.write_bytes(b"wav")
            text_path.write_text("hello", encoding="utf-8")
            output_path.write_text("existing", encoding="utf-8")

            with (
                patch("sys.argv", ["align.py", "--input", str(input_path), "--text", str(text_path), "--output", str(output_path)]),
                patch("builtins.input", return_value="n"),
                patch("align.AlignmentClient") as client_mock,
            ):
                align.main()

        client_mock.assert_not_called()

    def test_align_main_aligns_and_logs(self):
        import align

        base_config = types.SimpleNamespace(
            host="0.0.0.0",
            port=50051,
            auth_token="secret",
            job_wait_timeout_seconds=300.0,
            job_poll_interval_seconds=1.0,
        )
        fake_result = types.SimpleNamespace(
            words=(1, 2),
            language_code="en",
            srt_text="1\n00:00:00,000 --> 00:00:00,400\nhello world\n",
            srt_artifact_id="artifact-1",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "input.wav"
            text_path = root / "transcript.txt"
            output_path = root / "out.srt"
            input_path.write_bytes(b"wav")
            text_path.write_text("hello world", encoding="utf-8")
            channel_cm = MagicMock()
            channel_cm.__enter__.return_value = "channel"

            with (
                patch("sys.argv", [
                    "align.py", "--input", str(input_path), "--text", str(text_path), "--output", str(output_path), "--language", "en", "--device", "cpu", "--remove-punctuation",
                ]),
                patch("align.ServerConfig.from_sources", return_value=base_config),
                patch("align.grpc.insecure_channel", return_value=channel_cm) as insecure_channel_mock,
                patch("align.AlignmentClient") as client_mock,
            ):
                client_mock.return_value.align_file.return_value = fake_result
                align.main()

            self.assertEqual(output_path.read_text(encoding="utf-8"), fake_result.srt_text)
            insecure_channel_mock.assert_called_once_with("127.0.0.1:50051")
            client_mock.assert_called_once_with("channel", metadata=(("x-dictator-token", "secret"),))
            kwargs = client_mock.return_value.align_file.call_args.kwargs
            self.assertEqual(kwargs["transcript_file"], text_path)
            self.assertEqual(kwargs["language_code"], "en")
            self.assertTrue(kwargs["remove_punctuation"])

    def test_main_cli_handles_prompt_mp3_speech_timeline_and_cleanup(self):
        sf_module = types.SimpleNamespace(info=lambda path: types.SimpleNamespace(frames=24000, samplerate=24000))
        with patch.dict(sys.modules, {"soundfile": sf_module, "ffmpeg": types.SimpleNamespace()}):
            import main as main_module
            main_module = importlib.reload(main_module)

            result = types.SimpleNamespace(
                wav_paths=(Path("chunk1.wav"), Path("chunk2.wav")),
                segments=(types.SimpleNamespace(to_timeline_dict=lambda: {"content": "hello", "start": 0.0, "end": 1.0}),),
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                sample = root / "sample.mp3"
                text = root / "text.txt"
                out = root / "out.wav"
                speech = root / "speech.json"
                sample.write_bytes(b"mp3")
                text.write_text("Hello world", encoding="utf-8")

                fake_service = MagicMock()
                fake_service.synthesise_text.return_value = result
                with (
                    patch("sys.argv", ["main.py", "--sample", str(sample), "--text", str(text), "--output", str(out), "--speech", str(speech), "--sample-text", "Reference sample transcript", "--force"]),
                    patch("dictator.audio.ffmpeg_ops.mp3_to_wav") as mp3_to_wav,
                    patch("dictator.audio.ffmpeg_ops.concat_normalise") as concat_normalise,
                    patch("dictator.synthesis.service.SpeechSynthesisService", return_value=fake_service),
                    patch("dictator.synthesis.service.cleanup_synthesis_result") as cleanup_synthesis_result,
                    patch("dictator.transcription.service.transcribe_words", return_value=[{"content": "hello", "start": 0.0, "end": 1.0}]),
                ):
                    main_module.main()

                mp3_to_wav.assert_called_once()
                concat_normalise.assert_called_once()
                cleanup_synthesis_result.assert_called_once_with(result)
                timeline = json.loads(speech.read_text(encoding="utf-8"))
                self.assertEqual(timeline["textSegments"][0]["content"], "hello")
                self.assertEqual(timeline["voices"][0]["id"], "sample")

    def test_main_cli_rejects_empty_reference_audio_and_declines_overwrite(self):
        sf_module = types.SimpleNamespace(info=lambda path: types.SimpleNamespace(frames=0, samplerate=24000))
        with patch.dict(sys.modules, {"soundfile": sf_module, "ffmpeg": types.SimpleNamespace()}):
            import main as main_module
            main_module = importlib.reload(main_module)

            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                sample = root / "sample.wav"
                text = root / "text.txt"
                out = root / "out.wav"
                sample.write_bytes(b"wav")
                text.write_text("Hello", encoding="utf-8")
                out.write_text("existing", encoding="utf-8")

                with (
                    patch("sys.argv", ["main.py", "--sample", str(sample), "--text", str(text), "--output", str(out), "--sample-text", "Reference sample transcript"]),
                    patch("builtins.input", return_value="n"),
                ):
                    main_module.main()

                with patch("sys.argv", ["main.py", "--sample", str(sample), "--text", str(text), "--output", str(root / "fresh.wav"), "--sample-text", "Reference sample transcript"]):
                    with self.assertRaisesRegex(RuntimeError, "is empty"):
                        main_module.main()

    def test_main_cli_returns_early_when_synthesis_produces_no_wavs(self):
        sf_module = types.SimpleNamespace(info=lambda path: types.SimpleNamespace(frames=24000, samplerate=24000))
        with patch.dict(sys.modules, {"soundfile": sf_module, "ffmpeg": types.SimpleNamespace()}):
            import main as main_module
            main_module = importlib.reload(main_module)

            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                sample = root / "sample.wav"
                text = root / "text.txt"
                out = root / "out.wav"
                sample.write_bytes(b"wav")
                text.write_text("Hello", encoding="utf-8")

                fake_service = MagicMock()
                fake_service.synthesise_text.return_value = types.SimpleNamespace(
                    wav_paths=(),
                    segments=(),
                    temp_dir=root / "tmp-tts",
                )
                with (
                    patch("sys.argv", ["main.py", "--sample", str(sample), "--text", str(text), "--output", str(out), "--sample-text", "Reference sample transcript", "--force"]),
                    patch("dictator.synthesis.service.SpeechSynthesisService", return_value=fake_service),
                    patch("dictator.audio.ffmpeg_ops.concat_normalise") as concat_normalise,
                ):
                    main_module.main()

            concat_normalise.assert_not_called()

    def test_main_requires_sample_text_and_uses_qwen3_synthesis_request(self):
        sf_module = types.SimpleNamespace(info=lambda path: types.SimpleNamespace(frames=24000, samplerate=24000))
        with patch.dict(sys.modules, {"soundfile": sf_module, "ffmpeg": types.SimpleNamespace()}):
            import main as main_module
            main_module = importlib.reload(main_module)

            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                sample = root / "sample.wav"
                text = root / "text.txt"
                out = root / "out.wav"
                sample.write_bytes(b"wav")
                text.write_text("One. Two?", encoding="utf-8")

                with patch("sys.argv", ["main.py", "--sample", str(sample), "--text", str(text), "--output", str(out)]):
                    with self.assertRaises(SystemExit):
                        main_module.main()

                fake_service = MagicMock()
                fake_service.synthesise_text.return_value = types.SimpleNamespace(wav_paths=(), segments=(), temp_dir=root / "tmp")
                with (
                    patch("sys.argv", [
                        "main.py",
                        "--sample",
                        str(sample),
                        "--text",
                        str(text),
                        "--output",
                        str(out),
                        "--sample-text",
                        "Reference sample transcript",
                        "--force",
                    ]),
                    patch("dictator.synthesis.service.SpeechSynthesisService", return_value=fake_service),
                ):
                    main_module.main()

                request = fake_service.synthesise_text.call_args.args[0]
                self.assertEqual(request.engine, main_module.SynthesisEngine.QWEN3)
                self.assertEqual(request.speaker_transcript_text, "Reference sample transcript")
                self.assertEqual(request.text, "One. Two?")

    def test_extract_main_handles_decline_success_and_empty_transcription(self):
        torch_stub = types.SimpleNamespace(
            backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
                cudnn=types.SimpleNamespace(allow_tf32=False),
            ),
            device=lambda *args, **kwargs: None,
            cuda=types.SimpleNamespace(is_available=lambda: False),
        )
        librosa_stub = types.SimpleNamespace(feature=types.SimpleNamespace(rms=lambda y: __import__("numpy").array([[0.0]])))
        with patch.dict(sys.modules, {"torch": torch_stub, "librosa": librosa_stub, "ffmpeg": types.ModuleType("ffmpeg")}):
            import extract as extract_module
            extract_module = importlib.reload(extract_module)

            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                input_path = root / "input.wav"
                output_path = root / "output.wav"
                input_path.write_bytes(b"wav")
                output_path.write_text("existing", encoding="utf-8")

                with (
                    patch("sys.argv", ["extract.py", "--input", str(input_path), "--output", str(output_path)]),
                    patch("builtins.input", return_value="n"),
                    patch.object(extract_module, "run_with_timeout") as timeout_mock,
                ):
                    with self.assertRaises(SystemExit) as exc:
                        extract_module.main()
                    self.assertEqual(exc.exception.code, 0)
                    timeout_mock.assert_not_called()

                def fake_timeout(_timeout, name, func, *args, **kwargs):
                    if name == "decode":
                        return __import__("numpy").array([0] * extract_module.SAMPLE_RATE * 4, dtype=__import__("numpy").int16)
                    if name == "transcription":
                        kwargs["progress_cb"](2.0)
                        kwargs["progress_cb"](4.0)
                        return [{"content": "hello", "start": 0.5, "end": 1.0}]
                    if name == "trim":
                        return ("-3.0", 2.0)
                    raise AssertionError(name)

                with (
                    patch("sys.argv", ["extract.py", "--input", str(input_path), "--output", str(root / "new.wav"), "--force"]),
                    patch.object(extract_module, "run_with_timeout", side_effect=fake_timeout),
                    patch.object(extract_module, "load_diarization_pipeline", return_value="pipeline"),
                    patch.object(extract_module, "load_whisper_model", return_value="model"),
                    patch.object(extract_module, "apply_diarization_filter", return_value=[{"content": "hello", "start": 0.5, "end": 1.0}]),
                    patch.object(extract_module, "choose_window", return_value=0.0),
                    patch.object(extract_module, "compute_trim_bounds", return_value=(0.0, 1.5)),
                    patch.object(extract_module.logging, "info") as logging_info,
                ):
                    extract_module.main()
                self.assertTrue(any(call.args and call.args[0] == "  progress %d %%" for call in logging_info.call_args_list))

                with (
                    patch("sys.argv", ["extract.py", "--input", str(input_path), "--output", str(root / "empty.wav"), "--force"]),
                    patch.object(extract_module, "run_with_timeout", side_effect=[__import__("numpy").array([0], dtype=__import__("numpy").int16), []]),
                    patch.object(extract_module, "load_diarization_pipeline", return_value="pipeline"),
                    patch.object(extract_module, "load_whisper_model", return_value="model"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "no words transcribed"):
                        extract_module.main()

    def test_whisper_service_wrappers_delegate(self):
        import whisper_service

        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))}):
            with (
                patch("dictator.transcription.service.load_whisper_model", return_value="model") as load_mock,
                patch("dictator.transcription.service.transcribe_word_segments", return_value=["word"]) as segments_mock,
                patch("dictator.transcription.service.transcribe_words", return_value=[{"content": "hello"}]) as words_mock,
                patch("dictator.transcription.service.transcribe_text", return_value="hello") as text_mock,
            ):
                self.assertEqual(whisper_service.load_whisper_model("base"), "model")
                self.assertEqual(whisper_service.transcribe_word_segments("audio"), ["word"])
                self.assertEqual(whisper_service.transcribe_words("audio"), [{"content": "hello"}])
                self.assertEqual(whisper_service.transcribe_text("audio"), "hello")

        load_mock.assert_called_once_with("base")
        segments_mock.assert_called_once()
        words_mock.assert_called_once()
        text_mock.assert_called_once()

    def test_dictate_main_prints_words_and_default_payload(self):
        import dictate

        base_config = types.SimpleNamespace(
            host="0.0.0.0",
            port=50051,
            auth_token="secret",
            job_wait_timeout_seconds=300.0,
            job_poll_interval_seconds=1.0,
        )
        fake_result = types.SimpleNamespace(text="hello", words=[{"content": "hello"}], to_http_payload=lambda: {"text": "hello"})

        channel_cm = MagicMock()
        channel_cm.__enter__.return_value = "channel"
        with (
            patch("sys.argv", ["dictate.py", "--input", "audio.wav", "--include-words"]),
            patch("dictate.ServerConfig.from_sources", return_value=base_config),
            patch("dictate.grpc.insecure_channel", return_value=channel_cm),
            patch("dictate.DictationClient") as client_cls,
            patch("pathlib.Path.read_bytes", return_value=b"audio"),
            patch("builtins.print") as print_mock,
        ):
            client_cls.return_value.dictate_file.return_value = fake_result
            dictate.main()
        payload = json.loads(print_mock.call_args.args[0])
        self.assertEqual(payload["text"], "hello")
        self.assertIn("words", payload)

        with (
            patch("sys.argv", ["dictate.py", "--input", "audio.wav", "--language", "en"]),
            patch("dictate.ServerConfig.from_sources", return_value=base_config),
            patch("dictate.grpc.insecure_channel", return_value=channel_cm),
            patch("dictate.DictationClient") as client_cls,
            patch("pathlib.Path.read_bytes", return_value=b"audio"),
            patch("builtins.print") as print_mock,
        ):
            client_cls.return_value.dictate_file.return_value = fake_result
            dictate.main()
        payload = json.loads(print_mock.call_args.args[0])
        self.assertEqual(payload, {"text": "hello"})


if __name__ == "__main__":
    unittest.main()
