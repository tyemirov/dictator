# Dictator Go gRPC Contract

This module contains the generated Go protobuf and gRPC bindings for the
authoritative `dictator.speech.v1` contract owned by the Dictator service.

The `.proto` files in `proto/dictator/speech/v1/` are the source of truth.
Consumers should not hand-edit the generated Go files in this module.

The module also ships a small handwritten `Client` helper for the common
artifact-first flow. It keeps callers on the shared contract by uploading
complete media as an artifact before invoking speech RPCs such as
`TranscriptionService.Transcribe`.

To regenerate the contract artifacts from the checked-out proto sources:

```bash
make proto
```
