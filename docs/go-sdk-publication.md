# Go SDK publication

## Declaration

The application manifest declares one Go module as resource `go-sdk`.
The SDK version is independent of the application version.

| Field | Value |
| --- | --- |
| Source | `sdk/go/dictatorspeechv1` |
| Module | `github.com/tyemirov/dictator/sdk/go/dictatorspeechv1` |
| Declared version | `v1.12.0` |
| Module tag | `sdk/go/dictatorspeechv1/v1.12.0` |

Version `v1.12.0` adds native input measurements for LLM Proxy F070.
The previous published SDK version is `v1.11.0`.
Nine response messages retain exact sample count, sample rate, and measurement presence.
The SDK also retains the synthesis fields required by F042.
The declaration selects `v1.12.0` for publication. It does not prove that publication occurred.

## Lifecycle

Gateway F011 owns module validation, release evidence, and automatic tag publication.
Release seals the SDK source tree, module path, version, module tag, and target commit.
Publication creates or verifies the module tag from that sealed record.
An exact retry uses the existing publication receipt.

An application change can reuse the SDK version when the SDK source tree is unchanged.
Changed SDK source requires a new semantic version.
Gateway rejects changed source under an existing version before publication.
Deployment records the published SDK identity.

The existing application lifecycle also releases and publishes the declared GPU image.
Production deployment remains an operator action.
Use the [application lifecycle procedure](../README.md#gpu-release-flow) for the complete operation.

Before publication, merge the declaration into the application default branch.
Record the accepted Gateway revision after F011 and B540 validation.
Use the exact committed application and Gateway sources for publication.

## Validation

Run the local SDK and lifecycle checks:

```bash
make test-sdk-publication
```

This target tests the Go SDK and builds a separate consumer from the local SDK source.
The consumer verifies all nine input-usage responses through protobuf serialization.
It also verifies the existing synthesis fields.
The target then runs the sibling Gateway module tests through the public lifecycle.
Those tests cover initial publication, exact retry, unchanged-source reuse, changed-source rejection, module identity, and release evidence.
Gateway owns those generic test scenarios. Their local results are separate from Dictator publication evidence.

Run the repository checks before publication:

```bash
make ci
```

Go stub generation requires the declared generators: `protoc-gen-go@v1.36.11` and `protoc-gen-go-grpc@v1.6.2`.

After publication, verify the released SDK:

```bash
make verify-released-sdk
```

This command reads the declared module and version from the application manifest.
It uses a new Go module cache to retrieve that version through the Go package manager.
It builds the separate consumer against the released dependency and verifies the input quantities and synthesis fields.
The JSON result contains the module, version, Go download evidence, and field values.

## Publication evidence

Record these results for LLM Proxy F070 after publication:

- The accepted application and Gateway commits.
- The release receipt path and digest.
- The SDK artifact fields: `source`, `source_object`, `module`, `version`, `tag`, and `git_commit`.
- The publication receipt path and digest.
- The output from `make verify-released-sdk`.

The application Git directory contains lifecycle records under `mprlab-lifecycle`.
Locate that directory with `git rev-parse --path-format=absolute --git-path mprlab-lifecycle`.
Release receipts use `releases/<application-commit>/receipt.json` beneath that directory.

F002 remains open until the publication receipt and released consumer check succeed.
