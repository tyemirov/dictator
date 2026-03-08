# Genesis voice clone demo

This example serves a small browser UI that records a short voice sample, sends it to a running Dictator gRPC service, extracts a reference speaker sample, synthesizes a Genesis excerpt, and returns the resulting WAV file to the browser.

## Run it

Start Dictator first, then run:

```bash
python -m demo.voice_clone_web.app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` in a browser with microphone access.

For the Dockerized demo stack, serve the page over HTTPS so microphone access also works on non-localhost hosts. The repository wrapper does that with `ghttp` plus a local TLS certificate and key:

```bash
./scripts/voice-clone-demo.sh --tls-cert /path/to/computercat-cert.pem --tls-key /path/to/computercat-key.pem
```

## Expected inputs

- No browser-side auth token is required. The backend bridge reads `DICTATOR_GRPC_AUTH_TOKEN` from its environment.

The browser does not choose the Dictator gRPC target. In Docker, the backend bridge talks to the internal alias `dictator-grpc:50051`, which is provided by the active Dictator service profile.

The page asks the user to read:

> The quick brown fox jumped over the lazy dog. Eleven benevolent elephants balanced on bright blue bicycles.

It then asks Dictator to read a King James Version Genesis excerpt back in the user's voice.
