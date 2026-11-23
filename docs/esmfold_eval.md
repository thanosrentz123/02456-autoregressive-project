# Folding and visualising your samples on the HPC

How to take a CSV of generated sequences, fold it with ESMFold, and get numbers and pictures out.
Written after doing this for Phase 3 — including the traps.

Everything runs from the repository root.

---

## 0. One-time setup

```bash
bash hpc/setup_env.sh
```

Installs the environment and downloads the ESMFold weights (~8 GB) into `.cache/huggingface`
or `/work3/$USER`. Only needed once per checkout. Check it worked:

```bash
du -sh .cache/huggingface      # should be ~8 GB
```

If the weights are already there, skip this — the download is the slow part.

---

## 1. What you need

**One CSV with a `sequence` column.** That's the whole contract. Column of plain
amino-acid strings, anything else in the file is carried through untouched.

```csv
sequence,family
MNQTIQVIVSGHVQGVGFRMFTRKMANKLNLNGYVRNLPDG...,AcyP
```

Put it in `results/sequences/<name>.csv` (or anywhere — you pass the path).

**Use 256 sequences** if you can. Some metrics depend on set size, and the control sets are
generated at 256 by default, so anything else makes the comparison unfair.

---

## 2. Fold it

### The simple way — one set, interactive GPU

```bash
voltash                      # or sxm2sh; any GPU node
cd /path/to/02456-autoregressive-project
bash hpc/run_eval.sh results/sequences/my_samples.csv
```

Output goes to `results/eval/my_samples/`. To send it somewhere else, pass a second argument:

```bash
bash hpc/run_eval.sh results/sequences/my_samples.csv results/phase1/eval/my_samples
```

Budget **~4 minutes per 256 sequences**, plus a model load each time you invoke it.

### The batch way — several sets at once

`hpc/eval.sh` loops over a hardcoded list. **It currently points at `results/phase3/`**, so
copy it for your own phase rather than editing it in place:

```bash
cp hpc/eval.sh hpc/eval_phase1.sh
```

Then in your copy, change the set list, the sample path and the output path:

```sh
SETS="control_real control_shuffled mlp_samples rnn_samples"

for NAME in $SETS; do
    SAMPLES="results/sequences/$NAME.csv"          # <- your directory
    ...
    bash hpc/run_eval.sh "$SAMPLES" "results/phase1/eval/$NAME"
done
```

Also fix the log paths at the top (`#BSUB -o hpc/logs/eval_phase1.out`) so your job doesn't
overwrite someone else's, and set the walltime — **~4 min per set, so 15 sets needs hours, not
the default `-W 2:00`**. Submit with:

```bash
bsub < hpc/eval_phase1.sh
bjobs
```

---

## ⚠️ 3. The trap that cost us an afternoon

The TAs' `hpc/eval.sh` originally had:

```sh
cd "$(dirname "$0")/.."
```

With `bsub < script.sh` the script arrives on **stdin**, so LSF spools it to a temp file and `$0`
is that temp path — not `hpc/eval.sh`. The job ends up in the wrong directory, every
`[ ! -f "$SAMPLES" ]` check fails, and you get:

```
== control_real: no results/sequences/control_real.csv, skipping
== mlp_samples: no results/sequences/mlp_samples.csv, skipping
Done.
Successfully completed.
```

**A successful-looking job that folded nothing, in one second.** It's already fixed on the
`phase3-cfg` branch:

```sh
cd "${LS_SUBCWD:-$PWD}"
```

If you branched before that, apply it. `hpc/run_eval.sh` has the same line but is **fine** — it's
invoked as `bash hpc/run_eval.sh ...` with a real path, so `$0` resolves properly. Don't touch it.

**Rule of thumb: an LSF job that finishes in seconds has failed.** Check the runtime, not the
"Successfully completed".

---

## 4. Make the controls first

pLDDT means nothing on its own. Generate both controls once and fold them alongside your sets:

```bash
uv run python eval/make_controls.py --num 256
```

| Control | What it is | Our Phase 3 numbers |
|---|---|---|
| `control_real` | real proteins from `data/test.csv` | pLDDT **90.2**, identity 0.70 |
| `control_shuffled` | the same proteins, residues permuted | pLDDT **40.1**, identity 0.25 |

So 90 = real protein, 40 = garbage with the right amino acid composition. Your samples land
somewhere between, and that's the only way to read them. For reference, our unguided transformer
scored **83.2**.

---

## 5. What comes out

`results/<your path>/<set>/`:

| File | Contents |
|---|---|
| `eval.csv` | one row per sequence: validity, nearest-training identity, uniqueness, `mean_plddt` |
| `summary.csv` | one row for the whole set — the numbers you put in the report |
| `pdb/` | one structure per sequence, pLDDT in the B-factor column |
| `figures/` | PNG + interactive HTML for the first 5 structures |

`summary.csv` columns:

- **`mean_plddt`** — ESMFold's confidence, 0–100. The one that matters.
- **`nearest_identity_median`** — similarity to the closest training protein. Compare against
  the real control (0.70): *higher* than that means you're reproducing training data.
- **`frac_valid`**, **`frac_unique`** — these will be 1.000 and that means nothing. `frac_valid`
  can't fail if your sampler only emits the 20 residues, and `frac_unique` only catches
  byte-identical duplicates, which never happen at ~95 residues. Don't cite them as evidence of
  diversity.

---

## 6. Pictures of individual structures

```bash
uv run --group eval python eval/plot_structure.py results/eval/my_samples/pdb \
    --limit 5 --out results/eval/my_samples/figures
```

Per structure you get:

- **`.png`** — Cα trace coloured by pLDDT plus a per-residue confidence plot. Works anywhere, put
  this in the report.
- **`.html`** — interactive 3D cartoon. **Download it to your own machine and open it in a
  browser**; it won't render over SSH.

Colours follow the AlphaFold convention: dark blue >90, light blue 70–90, yellow 50–70,
orange <50. A sample that's mostly orange spaghetti didn't fold.

Add `--no-html` if `py3Dmol` isn't installed and you only want the PNGs.

Point it at a single file instead of a directory for just one structure:

```bash
uv run --group eval python eval/plot_structure.py results/eval/my_samples/pdb/sample_0.pdb
```

---

## 7. Comparing sets structurally

Beyond mean pLDDT — TM-score to real folds, secondary structure, compactness, structural
diversity:

```bash
uv run --group eval python eval/compare_structures.py \
    --sets real=results/eval/control_real \
           shuffled=results/eval/control_shuffled \
           mine=results/eval/my_samples \
    --reference real \
    --out results/eval/comparison
```

**List the real control first and name it as `--reference`** — its structures form the TM-score
panel everything else is measured against. TM > 0.5 usually means the same fold.

This is CPU-only and needs every set folded first.

---

## 8. Gotchas

| Problem | Cause / fix |
|---|---|
| Job "succeeds" in 1 second, folds nothing | the `cd` bug in §3 |
| `ModuleNotFoundError` on an eval script | needs `uv run --group eval`, not plain `uv run` |
| Job killed partway through the set list | walltime too low — ~4 min per 256-sequence set, plus a model load each. Completed sets are already on disk, so trim `SETS` and resubmit |
| `uv: command not found` in a job | `source hpc/env.sh` before any `uv` call |
| `summary.csv` exists but is empty | your CSV has a `setting` column containing the string `null` — pandas reads it as `NaN` and the grouping drops it. Use any other word |
| Home quota full | `source hpc/env.sh` keeps the caches out of `~`; run it before `uv` by hand too |
| `.html` shows nothing | open it locally, not over SSH |

---

## 9. Timings we measured

| Step | Cost |
|---|---|
| ESMFold weights download | once, ~8 GB |
| Folding 256 sequences | ~4 min on a V100, **plus a model load per invocation** |
| 15 sets × 256 | ~45 min total, 3,840 structures, **235 MB of PDB files** |
| Sequence metrics only (no `--fold`) | seconds, no GPU |

That last row is worth knowing: **drop `--fold` and everything except pLDDT runs on your laptop
in seconds.** Check your sequences look sane before spending GPU time on folding.

The PDB files are large and regenerable. Think before committing them — ours put 235 MB into the
repository history.
