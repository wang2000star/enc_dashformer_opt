# Engineering release — 23 September 2026

This release reopens enc_dashformer under the user's revised objective: useful
GitHub software first; evaluate a possible CCF-C paper only from defensible results.
Earlier CCF A/B project-closure notes do not stop engineering development.

## Changes in this release

- Promoted the existing CPU optimisation implementation into a standalone Go
  module. Unrelated iDASH research, papers, data and model files are excluded.
- Added `--data-dir` and `--check-inputs`, so the main executable can use externally
  supplied assets from any working directory without generating keys for preflight.
- Replaced process termination inside sequence/model readers with returned errors.
  Sequence parsing now handles whitespace and reports unknown tokens, inconsistent
  lengths, empty inputs, invalid tokenizer indices and scanner errors.
- Prevented a truncated feed-forward model from causing out-of-bounds slicing.
  Matrix parsing rejects empty, ragged and non-finite matrices.
- Reject incompatible baseline options early and stop on encrypted-evaluation
  errors before dereferencing an invalid output.
- Added a non-destructive build/run helper and paired benchmark runner with unique
  output directories, hashes, alternating execution order and failure records.
- Added English usage documentation, a multi-stage container definition, source
  attribution and a data-free GitHub Actions workflow.

Existing score/value compression, fixed-point masks and rotation optimisations
were carried forward; this release does not claim to have invented them or
establish a new speedup.

## Validation

- Full `go test -p 1 ./...`: passed, including encrypted rotation/key-switch and
  approximation tests on synthetic data.
- `go vet ./...`: passed.
- `go test -race ./utils`: passed. This is not a full inference race-detector run.
- Three Python regression tests: passed (output comparison, invalid scores and benchmark orchestration).
- Benchmark-runner integration test: passed; confirms alternating order, input
  preservation and refusal to overwrite an existing experiment.
- Binary build and local 100-example input/model preflight: passed. Preflight does
  not evaluate those examples or establish predictive accuracy.
- Shell syntax validation: passed.
- Container execution was not validated: the local Docker socket is inaccessible.
  The native binary was built and checked; CI execution is reported separately
  after publication rather than assumed from the workflow file.

## Completed end-to-end smoke check

Both paths completed on one locally available development example, producing
finite 25-class score vectors. With the same executable, input/model assets and
`--log-p 31,31`, the reference run took 313.853 seconds and the approximate
optimised run took 108.012 seconds. These are single-run functional measurements,
not repeated performance estimates or a throughput result.

The top prediction agreed, but maximum raw-score difference was 395.27145 and
score RMSE was 131.34689 after the legacy output scaling. These substantial score
differences must not be hidden behind the one-example prediction agreement.
No accuracy-preservation claim is established. Full evaluation must use matched
repeated runs and labelled data with an appropriate independent protocol.

Detailed logs, output files and the input/model/executable hash manifest remain
local in `results/engineering-smoke/`, excluded from publication. The smoke binary
was built before a formatting-only cleanup; executable hashes are in that local
manifest. No model or dataset is redistributed with this release.

## Limits and next engineering work

This remains a fixed-architecture, experimental implementation. Some legacy model
readers and numerical routines still assume model-specific dimensions/constants;
preflight is not comprehensive model validation. Internal parallelism is still
largely fixed at four workers. Low-rank defaults are approximations and need
application-specific evaluation. No new security certification, independent final
accuracy result, or CCF publication claim is made.

Upstream licensing is unresolved as described in NOTICE.md. This release publishes
source with attribution; it is not yet advertised as a fully licensed open-source
release. Resolve that status before making an unrestricted reuse promise.
