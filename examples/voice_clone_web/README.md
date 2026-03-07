# Genesis voice clone demo

This example serves a small browser UI that records a short voice sample, sends it to a running Dictator gRPC service, extracts a reference speaker sample, synthesizes a Genesis excerpt, and returns the resulting WAV file to the browser.

## Run it

Start Dictator first, then run:

```bash
python -m examples.voice_clone_web.app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` in a browser with microphone access.

## Expected inputs

- `Dictator gRPC URL`: `localhost:50051`, `grpc://localhost:50051`, or `https://your-host:443`
- `Auth Token`: the same token configured for Dictator
- `Language Code`: defaults to `en`

The page asks the user to read:

> The quick brown fox jumped over the lazy dog. Eleven benevolent elephants balanced on bright blue bicycles.

It then asks Dictator to read a King James Version Genesis excerpt back in the user's voice.
