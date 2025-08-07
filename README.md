# Dictator

## Archival-voice extraction and long-form voice-cloning

Two small, self-contained Python utilities:

| Script           | Purpose                                                                                                                                                                        |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`extract.py`** | Carve out the **clearest window** (default 20 s) from a noisy archival recording using Whisper ASR confidence + SNR heuristics, with a speaker-diarization model provided by `pyannote.audio`, then output a peak-normalised 24 kHz mono WAV. |
| **`main.py`**    | Feed that reference sample (or any WAV/MP3) to **[XTTS-v2]** and synthesise arbitrarily long speech from plain text – again to a peak-normalised 24 kHz mono WAV.              |

---

## 0 Prerequisites

You **must** run this project under **Python 3.11.8**, as some dependencies (pyannote.audio, Coqui-TTS) don’t ship
wheels for newer interpreters. We recommend installing via **pyenv**:

```bash
# 1) Install pyenv (if needed)
curl https://pyenv.run | bash
# Add these to your shell startup (~/.bashrc or similar):
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# 2) Install Python 3.11.8
pyenv install 3.11.8

# 3) Use 3.11.8 in this project directory
cd /path/to/dictator
pyenv local 3.11.8

# 4) Create & activate your venv
python3.11 -m venv .venv
source .venv/bin/activate

# 5) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
````

---

## 1 Quick start (GPU)

```bash
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
```

### CPU-only?

Both scripts fall back to `"cpu"` if CUDA is missing. `main.py` on CPU will be **slow** – consider shorter texts.

---

## 2 Folder layout

```text
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
```

Feel free to change folders – the scripts just take full paths.

---

## 3 `extract.py` usage

```text
usage: extract.py --input FILE --output FILE [options]

optional arguments
  --model {tiny,base,small,medium,large-v2,large-v3}
                        Whisper size (default: medium)
  --duration SECONDS    window length (default: 20)
  --min-confidence P    keep words whose P ≥ threshold (default: 0.80)
  --language CODE       Whisper language (e.g. 'en'); auto-detect if omitted
  --timeouts D T R      seconds for decode / transcribe / trim
  --force               overwrite existing output
```

**Algorithm**

1. FFmpeg → mono 16 kHz PCM
2. Whisper full-track transcription (progress heartbeat every 5 %)
3. Slide a fixed window; discard windows with spectral centroids outside
   `[500, 4_000]` Hz (`MIN_CENTROID_HZ`, `MAX_CENTROID_HZ`), then rank by
   **max words** and *(avg confidence × SNR)*
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

* Input text is **cleaned** (Unicode NFKC, whitespace collapsed).
* **Smart-split** into ≤ 800 bytes so XTTS never truncates mid-chunk.
* Synthesis stops when the next sentence would exceed `--length`.
* All chunks concatenated with FFmpeg, `dynaudnorm` + –1 dBFS, 24 kHz mono.

---

## 6 Dependencies

* **Python 3.11.8** (via pyenv; see “0 Prerequisites”)
* **FFmpeg** (with `dynaudnorm` filter) – e.g. `sudo apt install ffmpeg`
* Python libraries in `requirements.txt`

    * Torch wheels matching your CUDA version (see the [PyTorch site](https://pytorch.org/get-started/locally/))
    * ffmpeg-python
    * soundfile
    * numpy
    * openai-whisper
    * pyannote.audio
    * coqui-tts

---

## 7 License

Everything in this repo is released under the **MIT License**.
XTTS-v2 and Whisper licenses apply to their respective models.

Happy experimenting!
