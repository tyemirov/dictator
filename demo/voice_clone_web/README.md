# Voice clone demo

This example serves a small browser UI that records a short voice sample, sends it to a running Dictator gRPC service, converts the browser recording to WAV, synthesizes one of several preset passages from the full recorded sample, and returns the resulting WAV file to the browser.

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

> I grew up near a busy street, so even now I sleep best with a little noise in the distance.
> My friends know I speak quickly when I am excited, slow down when I am serious, and laugh before I finish the punch line.
> On cold mornings I want strong coffee, warm light, and ten quiet minutes to think.
> If you know me well, you can hear the difference between my polite voice, my tired voice, and the one I use when I am truly delighted.

It then asks Dictator to read back one of three preset passages in the user's voice: Genesis, the opening of Alice's Adventures in Wonderland, or Robert Frost's "Stopping by Woods on a Snowy Evening." The page can target XTTS, Qwen3-TTS, or CosyVoice 3.
