# enc_dashformer_opt

An engineering fork of [enc_dashformer](https://github.com/hangenba/enc_dashformer)
for CKKS-encrypted protein-sequence classification. It packages the existing
optimisation work with a configurable command-line interface, safer input parsing,
regression tests and reproducible local comparisons.

This is an experimental implementation of a fixed Dashformer architecture,
not a general Transformer runtime or a production privacy service. Publication
novelty, CCF ranking and accuracy preservation are not implied by this release.
See [provenance and licensing status](NOTICE.md) before redistribution.

## Included implementations

- Full-rank unfold/BSGS reference path (`--baseline`).
- Per-head score-rank approximation and optional value-basis compression.
- Fixed-point mask factorisation and optional shared-decomposition rotations.
- Explicit input/output locations, memory diagnostics and configurable BSGS width.
- Synthetic encrypted-kernel tests, without requiring competition data.

Low-rank and fixed-point options change numerical behaviour. Compare application
accuracy as well as runtime. Experimental rotation/rescale options remain opt-in;
`--help` describes the available controls.

## Build and test

Use Go 1.25 for the tested toolchain; the module declares Go 1.22.6.

```bash
git clone https://github.com/wang2000star/enc_dashformer_opt.git
cd enc_dashformer_opt
./reproduce.sh build
./reproduce.sh test
./build/enc-dashformer --help
```

The pinned Lattigo version is v5.0.0. Dependency upgrades require separate numerical
and API regression checks. Initial module downloads require network access.

## Supply your own model assets

The repository deliberately excludes competition datasets, model weights and
outputs. For the fixed supported architecture, supply:

```text
/path/to/assets/
  dashformer_tokenizer.json
  example_AA_sequences.list
  dashformer_model_parameters/
    embedding_Embedding_weights.txt
    positional_encoding_Lookup.txt
    transformer_block_Query_weights.txt
    transformer_block_Key_weights.txt
    transformer_block_Value_weights.txt
    transformer_block_CombineHead_weights.txt
    transformer_block_LayerNorm1_weights.txt
    transformer_block_LayerNorm2_weights.txt
    transformer_block_FFN_weights.txt
    layerNorm1_Reciprocal_SqrtVariance.txt
    layerNorm2_Reciprocal_SqrtVariance.txt
    Dense_Classifier_DenseClassifier_weights.txt
```

Each input row contains whitespace-separated amino-acid tokens, optionally followed
by a comma and a label. Rows must have the same length, matching the model's
positional encoding. Tokenizer IDs must be unique and in `1..vocabulary_size`;
channel zero is reserved. The existing model has 50 positions and 25 token channels.

Check readable input files and basic model/input dimensions without key generation:

```bash
./build/enc-dashformer --data-dir /path/to/assets --check-inputs
```

This is an input preflight, not a full model verifier. The numerical routines still
assume the original fixed architecture and model-specific approximation constants.

## Run

```bash
# Full-rank reference path
./build/enc-dashformer --data-dir /path/to/assets \
  --examples /path/to/sequences.list --output results/baseline \
  --baseline --log-p 31,31

# Existing approximate optimised path
./build/enc-dashformer --data-dir /path/to/assets \
  --examples /path/to/sequences.list --output results/optimized \
  --value-basis --hoisted-rotations --log-p 31,31
```

The default path uses score ranks `10,10,8,10`; `--value-basis` additionally enables
value ranks `24,24,24,24`. Direct runs overwrite files in the specified output
directory. Choose a fresh directory for each run. Runtime includes locally
simulated encryption and decryption; this is not a deployed client/server protocol.
The special-prime setting in these examples is an experimental configuration,
not a blanket security certification. Inference may require substantial RAM.

For alternating baseline/optimised runs without modifying inputs or source:

```bash
python3 scripts/benchmark.py --data-dir /path/to/assets \
  --examples /path/to/sequences.list --output-dir results/comparison --repeats 3
```

The runner refuses an existing output directory and records executable, input,
model and output hashes, commands, logs, exit status and median total runtime.
It does not calculate AUROC or claim accuracy preservation. Small development
subsets are not an independent official competition test set.

Compare the decrypted score files separately:

```bash
python3 scripts/compare_outputs.py results/comparison/01-baseline/output.txt \
  results/comparison/01-optimized/output.txt
```

This reports maximum absolute difference, RMSE and top-1 agreement, rejecting
mismatched or non-finite outputs. Agreement with the baseline is not accuracy
against ground-truth labels.

## Container

```bash
docker build -t enc-dashformer-opt .
docker run --rm -v /path/to/assets:/data:ro enc-dashformer-opt \
  --data-dir /data --check-inputs
```

For inference, mount a writable results directory and pass `--output` explicitly.
The image excludes local datasets and build outputs.

## Scope and validation

See [the engineering release record](docs/ENGINEERING_RELEASE.md) for checks run
and limitations. Other iDASH investigations and locally cached reference material
are maintained outside this repository. The next engineering priorities are full
model-shape validation, configurable worker limits, and matched repeated
end-to-end performance/accuracy measurements.
