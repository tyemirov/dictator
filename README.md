# Dictator
## Archival-voice extraction and long-form voice-cloning

Two small, self-contained Python utilities:

| Script           | Purpose                                                                                                                                                                        |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`extract.py`** | Carve out the **clearest window** (default 20 s) from a noisy archival recording using Whisper ASR confidence + SNR heuristics, then output a peak-normalised 24 kHz mono WAV. |
| **`main.py`**    | Feed that reference sample (or any WAV/MP3) to **[XTTS-v2]** and synthesise arbitrarily long speech from plain text – again to a peak-normalised 24 kHz mono WAV.              |

---

## 1 Folder layout

```

dictator/
├── assets/
│   ├── sources/
│   │   ├── voices/      # original long recordings
│   │   └── texts/       # .txt books / speeches / etc.
│   └── samples/         # reference clips cut by extract.py
├── output/              # final generated audio
├── extract.py
├── main.py
├── requirements.txt
└── README.md            # ← you are here

````

Feel free to change folders – the scripts just take full paths.

---

## 2 Quick start (GPU)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # Torch CUDA 11.8/12.1 wheels

# 🔈 1) cut a 20-second Churchill clip
python extract.py \
       --input  assets/sources/voices/churchill_disc1side2.flac \
       --output assets/samples/churchill.wav

# 📖 2) clone the voice and read Genesis, capped at 3 minutes
python main.py \
       --sample assets/samples/churchill.wav \
       --text   assets/sources/texts/genesis.txt \
       --output output/genesis_by_churchill.wav \
       --length 3m
````

### CPU-only?

Both scripts automatically fall back to `"cpu"` if CUDA is missing.
`main.py` on CPU will be **slow** – consider shorter texts.

---

## 3 `extract.py` usage

```text
usage: extract.py --input FILE --output FILE [options]

optional arguments
  --model {tiny,base,small,medium,large-v2,large-v3}
                        Whisper size (default: medium)
  --duration SECONDS    window length (default: 20)
  --min-confidence P    keep words whose P >= threshold (default: 0.80)
  --timeouts D T R      seconds for decode / transcribe / trim
  --force               overwrite existing output
```

Algorithm

1. FFmpeg → mono 16 kHz PCM
2. Whisper full-track transcription (progress heartbeat every 5 %)
3. Slide a fixed window; rank by **max words** then *(avg confidence × SNR)*
4. FFmpeg lossless trim + peak-normalise to –1 dBFS, resample 24 kHz

Typical runtime on an RTX 3060 for a 30-minute 44 kHz FLAC is \~70 s.

---

## 4 `main.py` usage

```text
usage: main.py --sample WAV/MP3 --text TXT --output WAV [options]

optional arguments
  --length 10s|3m|1.5h   cap final audio; stops on last full sentence
  --language CODE       TTS language code (default: en)
  --force                overwrite existing output
```

* The input text is **cleaned** (Unicode NFKC, whitespace collapsed).
* It’s **smart-split** into ≤ 240-char chunks so XTTS never truncates mid-chunk.
* Synthesis stops when the next sentence would exceed `--length`.
* All chunks are concatenated with FFmpeg, `dynaudnorm` + –1 dBFS, 24 kHz mono.

---

## 5 Troubleshooting

| Symptom                             | Fix                                                                                                                                                                                                                                                                                             |
|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *Model download looks stuck at 0 %* | Some networks block **Git LFS**. </br>Run `curl -L https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin -o /dev/null` – if that stalls, use a VPN or pre-download models on another connection and copy them to `~/.cache/whisper` (Whisper) or `~/.cache/tts` (XTTS). |
| `cudnn*.so not found`               | CUDA works fine – `extract.py`/Whisper can run with Torch’s fallback kernels. Ignore unless you need peak GPU speed.                                                                                                                                                                            |
| `WARNING text length exceeds … 250` | Our splitter honours 240 chars, so you shouldn’t see this anymore.                                                                                                                                                                                                                              |

---

## 6 Dependencies

* Python 3.10 – 3.12
* **FFmpeg** (build with `dynaudnorm` filter) – `sudo apt install ffmpeg`
* The Python libs in `requirements.txt`

    * Torch wheel must match your CUDA version (see [PyTorch hub](https://pytorch.org/get-started/locally/))

---

## 7 License

Everything in this repo is released under the **MIT License**.
XTTS-v2 and Whisper licences apply to their respective models.

Happy experimenting!


