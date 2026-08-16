# ISSUES

Entries record newly discovered requests or changes.

Read @AGENTS.md (Workflow section), @POLICY.md, and relevant stack guides before implementing changes.

Format: `- [ ] [B042] (P1) {I007} Title`

- `[ ]` open, `[-]` taken, `[!]` blocked, `[x]` closed.
- Blocked issues (`[!]`) must include a `Blocked:` line in the body.

## BugFixes

- [ ] [B001] (P0) The hosted diarization route must return valid gRPC results.
  Goal:
  A valid authenticated `DiarizeAudio` request reached
  `dictator.mprlab.com:443`.
  The hosted route returned HTTP 504 without a gRPC content type.
  The client received no typed gRPC status and no diarization artifact.
  The route must return a valid gRPC result for each documented RPC.

  Requirements:
  - Reproduce operation `mediaops-20260816T065915-000001` with a test audio
    file.
  - Identify the service, route, or upstream deadline that produced HTTP 504.
  - Return a typed gRPC status when a synchronous request cannot complete.
  - Keep `SubmitDiarizeAudioJob`, `GetDiarizeAudioJob`, and
    `CancelDiarizeAudioJob` available through the hosted route.
  - Make the hosted job path complete without a plain HTTP response.
  - Preserve artifact identity and final job state across each status query.
  - Do not increase a timeout as the primary repair.

  Deliverables:
  - Repair the Dictator runtime or its declared hosted route contract.
  - Add a public-route regression for gRPC content type and typed status
    behavior.
  - Record a successful hosted job receipt and diarization artifact.

  Validation:
  - Verify hosted gRPC health and authenticated metrics.
  - Submit a live diarization job through the hosted TLS route.
  - Query the job until it reaches `DIARIZATION_JOB_STATE_SUCCEEDED`.
  - Download or read the persisted diarization artifact.
  - Run the MediaOps live canary with `mediaops.audio.speech.diarize`.
  - Run the Creative Director accepted-audio canary through `accepted-assets`.
  - Run `make ci`.

## Improvements

- [ ] [I001] (P1) {P001} Build the Higgs TTS 3 qualification harness
  Goal:
  Produce repeatable evidence for the adoption contract from P001.

  Requirements:
  - Add `scripts/qualify_higgs_tts3.py` as the qualification entrypoint.
  - Add a versioned qualification manifest without private audio.
  - Read all revisions, hashes, hardware values, workloads, and limits from the P001 contract.
  - Use the same consented reference samples for Qwen3 and Higgs TTS 3.
  - Measure intelligibility, speaker similarity, duration accuracy, startup time, latency, real-time factor, memory, and failure rate.
  - Do a test of cloning, each approved language, code switching, long text, reference reuse, timeout, and cancellation.
  - Run a 100-request soak with the concurrency value from P001.
  - Run the approved Whisper, WhisperX, diarization, and Silero workloads during the resource test.
  - Do a test of the approved sidecar placement with gateway-supported container options.
  - Write machine-readable results and a readable qualification report.

  Deliverables:
  - Add the qualification entrypoint, manifest schema, automated tests, and usage document.
  - Record exact commands, revisions, hashes, hardware facts, measurements, and output hashes.
  - Record one result for each P001 acceptance limit.
  - Record `approved` or `rejected` as the final result.

  Validation:
  - Reproduce all recorded samples from the qualification manifest.
  - Confirm that each accepted file is a 24 kHz mono WAV file.
  - Confirm that the report contains no omitted result or undefined limit.
  - Confirm that an approved result meets every P001 limit.
  - Run the qualification harness tests.

- [ ] [I002] (P1) {I001} Replace Qwen3 with the Higgs TTS 3 sidecar
  Goal:
  Replace the Qwen3 API and runtime in one forward-only change.

  Requirements:
  - Start this issue only when I001 records `approved`.
  - Use the model, runtime, container, settings, limits, and placement from P001 and I001.
  - Reserve protobuf number `1` and the name `SYNTHESIS_ENGINE_QWEN3`.
  - Add `SYNTHESIS_ENGINE_HIGGS_TTS3` with protobuf number `3`.
  - Add `HIGGS_TTS3` with the internal value `higgs_tts3`.
  - Regenerate the checked-in Python and Go SDK files.
  - Route reference-audio synthesis to Higgs TTS 3.
  - Keep Russian preset-speaker synthesis on Silero.
  - Require `speaker_artifact_id` and `speaker_transcript_text` for Higgs voice cloning.
  - Implement `HiggsTTS3Backend` with the approved sidecar HTTP contract.
  - Apply the approved timeout, text, sampling, and inflight limits.
  - Propagate gRPC deadlines and job cancellation to the sidecar request.
  - Validate the sidecar identity, response status, error body, and WAV properties.
  - Map sidecar failures to documented Dictator error codes.
  - Exclude text, reference audio, data URLs, and tokens from normal logs.
  - Build the sidecar image from the approved immutable revisions.
  - Verify all model hashes before the sidecar becomes ready.
  - Add the sidecar to `docker-compose.yml` with its required health check.
  - Preserve current artifact, duration, chunk, progress, timeline, and Silero behavior.
  - Remove all Qwen3 code, configuration, dependencies, images, assets, tests, and documentation.
  - Remove Qwen3 and its FlashAttention package from `Dockerfile.gpu`.
  - Keep the GPU packages that the other Dictator services require.
  - Update the CLI, browser demo, examples, README, and client documentation.

  Deliverables:
  - Add the Higgs protobuf, domain, routing, backend, configuration, and generated SDK contracts.
  - Add the immutable sidecar image definition and model verification data.
  - Add the two-service local Compose contract and image probes.
  - Add contract, adapter, cancellation, failure, image, and workflow tests.
  - Remove all active Qwen3 surfaces from the source tree.

  Validation:
  - Run `make ci`.
  - Do a test of the adapter with a controlled local HTTP service.
  - Do a test of direct synthesis and asynchronous synthesis jobs.
  - Do a test of defaults, explicit engine values, and rejected protobuf value `1`.
  - Do a test of timeout, cancellation, invalid audio, model mismatch, and an unavailable sidecar.
  - Do a test of artifacts, duration limits, progress, timelines, Silero, and Whisper.
  - Complete one real GPU voice-clone request through Dictator and the sidecar.
  - Scan active files for Qwen3 identifiers, dependencies, model assets, and configuration.

- [ ] [I003] (P1) Migrate the current Dictator lifecycle to schema v3
  Goal:
  Replace the schema-v1 production contract without a change to local orchestration.

  Requirements:
  - Keep `docker-compose.yml`, local scripts, `make up`, `make down`, and local tests.
  - Rewrite `.mprlab/deploy/resources.yml` with `schema_version: 3`.
  - Declare only the current Dictator image and its current production resources.
  - Preserve the public Dictator gRPC capability and Caddy route.
  - Bind private values through `.mprlab/deploy/.env` and typed private-value resources.
  - Delegate `make release`, `make publish`, and `make deploy` to the physical sibling gateway.
  - Remove `scripts/release.sh`, `scripts/publish-release.sh`, `scripts/deploy.sh`, and `scripts/release/`.
  - Replace their contract tests with gateway delegation tests.
  - Keep application source CI separate from gateway lifecycle validation.

  Deliverables:
  - Add the current schema-v3 Dictator application manifest.
  - Add canonical lifecycle delegation for one Dictator image.
  - Remove obsolete app-owned production lifecycle paths.
  - Preserve all app-owned local orchestration paths.

  Validation:
  - Run the gateway selected-manifest isolation check from the primary checkouts.
  - Confirm that the plan contains only Dictator-owned resources.
  - Confirm that the plan preserves the public gRPC route.
  - Run the local orchestration contract tests.
  - Run `make ci`.

- [ ] [I004] (P1) {I002,I003} Add Higgs TTS 3 to the schema-v3 lifecycle
  Goal:
  Add the approved two-image topology to the current production manifest.

  Requirements:
  - Use the placement and container options that I001 approved.
  - Declare immutable Dictator and Higgs TTS 3 image artifacts.
  - Declare both services, placements, GPU needs, volumes, ports, health checks, and dependency order.
  - Declare the private Higgs HTTP capability for Dictator.
  - Keep the existing public Dictator gRPC capability and Caddy route.
  - Add each required private value as a typed resource.
  - Seal both images in one application release definition.
  - Keep `docker-compose.yml` as the local orchestration contract.

  Deliverables:
  - Update `.mprlab/deploy/resources.yml` with the approved Higgs topology.
  - Add one release definition for both immutable images.
  - Add lifecycle tests for image identity, capability resolution, order, and health checks.

  Validation:
  - Run the gateway selected-manifest isolation check from the primary checkouts.
  - Confirm that the plan contains only Dictator-owned resources.
  - Confirm that the plan resolves the private Higgs capability.
  - Confirm that the plan preserves the public gRPC route.
  - Confirm that the release receipt fixture identifies both image digests.
  - Run `make ci`.

- [ ] [I005] (P1) {I004} Verify the Higgs TTS 3 candidate in a non-production environment
  Goal:
  Prove the replacement on the non-production host that P001 identifies.

  Requirements:
  - Deploy the exact candidate image digests through the schema-v3 lifecycle.
  - Record the host, GPU, driver, memory, storage, model, image, release, and gateway identities.
  - Do a test of authenticated upload, synthesis, job polling, download, duration, and timeline operations.
  - Do a test of Silero synthesis and all unchanged transcription services.
  - Run the I001 corpus and soak workload through the public non-production route.
  - Run the approved concurrent GPU workload.
  - Compare all measurements with the P001 limits.
  - File each reproducible contract failure as a separate BugFix issue.

  Deliverables:
  - Record the deployment receipt and exact runtime identities.
  - Record API, quality, performance, resource, and soak evidence.
  - Record `accepted` or `rejected` as the final result.

  Validation:
  - Confirm that the deployed digests match the release receipt.
  - Confirm that all authenticated API workflows pass.
  - Confirm that every P001 limit passes.
  - Confirm that the active containers contain no Qwen3 runtime or model asset.
  - Confirm that each found defect has a BugFix issue.

- [ ] [I006] (P1) {P002} Implement the bounded Qwen3 job migration
  Goal:
  Implement only the persisted-record action that P002 approves.

  Requirements:
  - Use the record selection, target state, retention rule, and storage path from P002.
  - Add a dry-run mode that changes no record.
  - Reject records outside the approved selection.
  - Preserve completed audio and timeline artifacts.
  - Write a receipt with counts, record identifiers, prior states, final states, and errors.
  - Make a repeated execution produce no additional record change.
  - Add fixture tests for each selected and rejected record state.

  Deliverables:
  - Add the bounded one-off migration command and its tests.
  - Add the dry-run report and receipt schema.
  - Document the exact execution and verification commands for I007.

  Validation:
  - Run the migration tests with production-shaped fixtures.
  - Confirm that dry-run output matches the selected fixture records.
  - Confirm that a second fixture run changes no record.
  - Run `make ci`.

- [!] [I007] (P1) {I005,I006,P002} Execute the production Higgs TTS 3 cutover
  Goal:
  Activate the accepted replacement through one authorized production operation.

  Requirements:
  - Use the cutover order, maintenance window, client gates, and recovery boundary from P002.
  - Confirm that every client gate from P002 is completed.
  - Release and publish the exact source revision that I005 accepted.
  - Verify both published image digests before deployment.
  - Run the I006 migration in dry-run mode.
  - Compare the dry-run receipt with the approved P002 selection.
  - Deploy through the schema-v3 gateway lifecycle.
  - Run the authenticated production acceptance workflow.
  - Execute the I006 migration only after runtime acceptance passes.
  - Remove the one-off migration command after a successful migration.
  - Record source, release, publication, deployment, image, model, gateway, and migration identities.

  Deliverables:
  - Record the sealed release and publication receipts.
  - Record production runtime, public API, and migration acceptance evidence.
  - Update release notes and durable operator documentation.
  - Remove the completed one-off migration path.

  Validation:
  - Confirm that production runs the accepted image and model identities.
  - Confirm that authenticated upload, synthesis, job, download, duration, and timeline operations pass.
  - Confirm that Silero and all transcription services pass.
  - Confirm that all client gates remain completed.
  - Confirm that the active runtime and source tree contain no Qwen3 path.
  - Confirm that the one-off migration command no longer exists.

  Blocked:
  - The production maintenance window, credentials, and deployment authorization are not recorded.

## Maintenance

## Features

- [ ] [F001] (P2) {I005,P003} Add explicit Higgs expressive speech controls
  Goal:
  Add the public control format that P003 defines.

  Requirements:
  - Add `SYNTHESIS_TEXT_FORMAT_HIGGS_CONTROL_TOKENS` to the protobuf contract.
  - Keep plain text as the default Higgs text format.
  - Reject Higgs control-token syntax in plain text.
  - Implement the exact grammar, token table, placement rules, and error codes from P003.
  - Implement the exact timeline rules from P003.
  - Update generated SDK files, examples, demo controls, and API documentation.

  Deliverables:
  - Add the explicit Higgs control-token format and parser.
  - Add the P003 token table for the approved model revision.
  - Add timeline-safe rendering for controlled text.
  - Add client examples for each supported control category.

  Validation:
  - Do a test of each P003 grammar rule and approved token value.
  - Do a test of each P003 invalid-input case and error code.
  - Do a test of each P003 timeline fixture.
  - Run the P003 listening corpus and confirm that all thresholds pass.
  - Run `make ci`.

## Planning

Planning issues do not authorize implementation.

- [ ] [P001] (P1) Approve the Higgs TTS 3 adoption contract
  Goal:
  Provide all fixed inputs for a reproducible Higgs TTS 3 qualification.

  Requirements:
  - Record the decision owner and source for each model, code, and data license conclusion.
  - Decide production hosting, image distribution, model distribution, attribution, and voice-cloning use.
  - Record the approved test host and its GPU, driver, CUDA, CPU, memory, and storage values.
  - Record the approved production placement and gateway-supported container options.
  - Pin the model, serving runtime, source, dependency, base image, and container revisions.
  - Record the SHA-256 value for each model and distributable artifact.
  - Define the sidecar HTTP request, response, health, identity, timeout, cancellation, and error contracts.
  - Define sampling, text, reference-audio, and global inflight limits.
  - Define the consented corpus, languages, workloads, sample count, and concurrency.
  - Define numeric limits for quality, latency, real-time factor, memory, startup, and failure rate.
  - Define the non-production host and its deployment authorization.
  - Select `approved` or `rejected` for qualification work.

  Deliverables:
  - Add `.mprlab/HIGGS_TTS3_ADOPTION.md` with all decisions and source evidence.
  - Add a machine-readable qualification contract without private audio or credentials.
  - Record each decision owner, decision date, source revision, and source location.

  Validation:
  - Confirm that every required field contains one value and no placeholder.
  - Confirm that every limit has a number, unit, sample size, and aggregation rule.
  - Confirm that every revision and model artifact has an immutable identity.
  - Confirm that the corpus records consent provenance without private audio.
  - Confirm that the selected topology uses current gateway container fields.

- [ ] [P002] (P1) {I005} Produce the production cutover plan
  Goal:
  Provide the owner-approved inputs for the production operation.

  Requirements:
  - Start this issue only when I005 records `accepted`.
  - Inventory each client that sends an explicit Qwen3 engine value.
  - Record one repository, owner, update issue, validation command, and completion gate for each client.
  - Inventory each persisted Qwen3 job state on the production target.
  - Decide the selected records, target state, retention rule, and storage path for one migration.
  - Define the migration dry run, receipt, stop condition, and verification commands.
  - Define the release revision, maintenance window, deployment owner, and credential owner.
  - Define predeployment, deployment, postdeployment, and recovery gates.
  - Define the exact authority that can approve production activation.

  Deliverables:
  - Add one cutover plan with the complete client inventory and operation order.
  - Add the approved one-off migration specification.
  - Record all owners, commands, gates, receipts, and stop conditions.

  Validation:
  - Confirm that each client has a completed update gate or no Qwen3 dependency.
  - Confirm that the migration specification selects a bounded record set.
  - Confirm that no step requires an unrecorded value or decision.
  - Confirm that each production action has an owner and evidence requirement.

- [ ] [P003] (P2) {I001} Define the Higgs expressive control contract
  Goal:
  Provide all fixed inputs for F001 without a change to the baseline replacement.

  Requirements:
  - Use the exact model revision that I001 approves.
  - List each supported control category, token, value, placement rule, and incompatible combination.
  - Define the plain-text rejection rule and each stable error code.
  - Define the timeline result for delivery, pause, sound, and written-cue controls.
  - Define the public grammar without hidden model tokens or undocumented behavior.
  - Define the listening corpus, evaluators, scoring method, and numeric acceptance limits.

  Deliverables:
  - Add one versioned control grammar and token table.
  - Add valid, invalid, and timeline fixtures for every grammar rule.
  - Add one listening-test specification with numeric limits.

  Validation:
  - Confirm that each grammar production has valid and invalid fixtures.
  - Confirm that each token maps to the approved model revision.
  - Confirm that each invalid case maps to one error code.
  - Confirm that each listening limit has a number, unit, sample size, and scoring rule.
