# Autoregressive models of protein sequences

A 02456 Deep Learning project. You build autoregressive sequence models - an MLP with a fixed
context, an RNN / LSTM / GRU, a transformer - on one frozen train/validation/test split of three
protein families, and then measure what they generate.

What the repository gives you is the two ends of that work:

- **the data and a dataset class** (`data/`, `src/protgen/data/`), so every model is fitted and
  compared on the same split and the same tokenisation;
- **an evaluation pipeline** (`eval/`, `src/protgen/metrics/`), so "the samples look like
  proteins" becomes a number you can put in a table.

**The middle is yours**: the models, the training loop and the sampling code. `src/protgen/models/`
and `config/` are empty placeholders waiting for them.

## Get the code

```bash
git clone https://github.com/Christianhvilshoej/02456-autoregressive-project.git
cd 02456-autoregressive-project
```

Everything below is run from that directory. The data is in the repository, so the clone is all
you need - there is nothing to download separately.

On the cluster, clone it somewhere with room rather than into your home directory, which has a
small quota; see [On the cluster](#on-the-cluster-dtu-hpc).

To work on your own copy, fork it on GitHub first and clone the fork, or point this clone at your
own repository:

```bash
git remote set-url origin https://github.com/<your-user>/<your-repo>.git
```

## Install

The project is managed with [uv](https://docs.astral.sh/uv/). uv installs Python itself as well as
the dependencies, so it is the only thing you need on the machine beforehand:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

`brew install uv`, `pipx install uv` and `winget install --id=astral-sh.uv` work too. Open a new
shell afterwards so that `uv` is on the PATH, then, from the project root:

```bash
uv sync
```

That reads `pyproject.toml` and `uv.lock`, fetches Python 3.13 if the machine has no suitable
interpreter, and creates `.venv/` with the exact locked versions. Prefix a command with `uv run` to
use that environment without activating it:

```bash
uv run python eval/eval.py --samples results/sequences/mock_data.csv
```

If that prints a small table, the environment works. (The project sets `link-mode = "copy"`,
because a checkout inside a OneDrive folder cannot take uv's default hardlinks. On the cluster
`hpc/env.sh` asks for hardlinks again, where they work and where the environment is large enough
for the difference to matter: about 1 GB on Linux without a GPU, 9.5 GB with the CUDA build of
torch.)

`uv sync` also installs the `protgen` package itself, in editable mode. `import protgen` then works
from anywhere - a notebook in `notebooks/`, a script on the cluster, the project root - with no
`sys.path` lines. The package lives in `src/protgen/`, the src layout, so `import protgen` can only
ever resolve to the installed package and never to a directory that happens to sit in the current
working directory.

**The `eval` group is separate.** ESMFold and its dependencies are only needed for folding, so they
are an optional dependency group:

```bash
uv sync --group eval        # install them
uv run --group eval ...     # install them if missing, then run
```

Mind the difference between the two: `uv run --group eval ...` installs the group if it is missing,
while a plain `uv sync` *removes* it again, because it syncs the environment to the default groups
only. The scripts in `eval/` therefore pass `--group eval` on every call, and `uv sync --group eval`
is the one to re-run after pulling.

## The data

Three protein families, balanced, in `data/train.csv`, `data/val.csv` and `data/test.csv` - one row
per protein, columns `sequence,family`. The split is frozen and the files share no sequence, so the
test split is a genuine held-out set; read it once, at the end, and never for model selection.

| Split | Proteins | Per family | Length (min / mean / max) |
|---|---|---|---|
| `train` | 16,020 | 5,340 | 68 / 95.0 / 127 |
| `val` | 2,001 | 667 | 68 / 94.6 / 126 |
| `test` | 2,001 | 667 | 68 / 94.2 / 127 |

The three families are matched in length, so **length carries no information about the family** -
useful to know when a classifier of yours does suspiciously well.

### `ProteinDataset`

`protgen.data.ProteinDataset` is an ordinary `torch.utils.data.Dataset` over one of those CSVs. It
tokenises every sequence once, in `__init__`, and serves both of the project's targets from the
same `x`:

```python
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset

train_data = ProteinDataset("data/train.csv")
val_data = ProteinDataset("data/val.csv")
print(train_data)
# ProteinDataset(num_samples=16020, vocab_size=23, num_classes=3, max_length=129)

x, x_shifted, y, length = train_data[0]
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

for x, x_shifted, y, length in train_loader:
    ...
```

| Item | Shape | What it is |
|---|---|---|
| `x` | `(129,)` long | `<BOS>`, the residues, `<EOS>`, then `<PAD>` to `max_length` |
| `x_shifted` | `(129,)` long | `x` shifted left by one: the next-token target, with `IGNORE_INDEX` (`-1`) past the end of the sequence |
| `y` | scalar long | the family, as a class index |
| `length` | int | the number of residues, without `<BOS>`/`<EOS>`/`<PAD>` |


Whole-split tensors are the same data, if a small model is easier to write against them than
against a loader:

```python
X = train_data.tokens            # (16020, 129) long
Y_gen = train_data.tokens_shifted  # (16020, 129) long, -1 at the padding
Y_fam = train_data.labels          # (16020,) long
```

**The vocabulary** is fixed in `protgen.data`, not inferred from the data, so a token id means the
same residue in every run and checkpoint: `<PAD>` 0, `<BOS>` 1, `<EOS>` 2, then the 20 standard
residues in alphabetical order, 3 to 22. `train_data.vocab` is that mapping, `vocab_size` is 23,
and `encode` / `decode` go between a residue string and a token tensor (`decode` drops the special
tokens, so `decode(x)` and `decode(x_shifted)` both give the protein back).

**The family ids** are fixed the same way, in `protgen.data.FAMILIES`:

| `y` | family |
|---|---|
| 0 | AcyP |
| 1 | HPr |
| 2 | HU |

`train_data.fam_to_idx` is that mapping. `notebooks/introduction.ipynb` runs through all of the
above in a few cells.

## What you build

| Where | What goes there |
|---|---|
| `src/protgen/models/` | the architectures. `mlp.py` is an empty placeholder; add the recurrent and transformer models beside it. Anything you import as `protgen.models...` is picked up without reinstalling, because the package is installed in editable mode. |
| `config/` | run configuration, if you want it. [Hydra](https://hydra.cc) is already a dependency, so `config/default.yaml` can grow into a config group tree; the file is empty, and nothing reads it yet. |
| the project root | your `train.py` / `generate.py`, or notebooks, whichever you prefer. |
| `hpc/train.sh` | the queue, walltime and memory for the training job, and the command that runs it. Fill in the command at the bottom. |

The one interface worth keeping to is the output of your sampling code: **a CSV with a `sequence`
column**, written to `results/sequences/<name>.csv`. That is all `eval/` asks for, which is why
nothing in `eval/` has to know how your model works.

## Evaluating what you generate

`eval/` scores a set of sequences on three questions - is it a protein, is it new, is it the same
one again - and, on a GPU, folds it with ESMFold and reports the confidence. Two control sets, real
test proteins and the same proteins with their residues shuffled, fix the scale, because none of
those numbers means much read on its own.

The smoke test under [Install](#install) is already that pipeline in full, run on the mock sample
set that ships with the repo. Point it at your own CSV instead, add `--fold` on a GPU node, and
read what comes out against the controls. **[`eval/README.md`](eval/README.md)** is the one place
that describes the scripts, their output and what each metric means.

## On the cluster (DTU HPC)
If you a new to the DTU HPC look at this guide: 
https://docs.google.com/document/d/1bRtX87ZD7faG1b5CayN5LKUPcn75xmrWt6jFcuwoGMU/edit?tab=t.0

If you do not have a scratch directory on the hpc (eg. /work3), you need to either request access (follow the guide) or wait until Jes has requested storage for the whole course. 

Clone the repo somewhere with room rather than into your home directory, whose quota is small:

```bash
cd /work3/$USER      # or anywhere else with space
git clone https://github.com/Christianhvilshoej/02456-autoregressive-project.git
cd 02456-autoregressive-project
```

**Everything the cluster needs lives in `hpc/`**, and nothing in `eval/` is specific to it: the
Python entry points run anywhere, the shell scripts in `hpc/` are how you run them here.

| In `hpc/` | How you run it | What it is |
|---|---|---|
| `setup_env.sh` | `bash hpc/setup_env.sh` | one-time setup on the login node: installs uv and the environment, downloads the ESMFold weights (~8 GB) |
| `train.sh` | `bsub < hpc/train.sh` | your training run. Fill in the command at the bottom |
| `eval.sh` | `bsub < hpc/eval.sh` | scores every sample set listed at the top of the file - **add your own there** |
| `run_eval.sh` | `bash hpc/run_eval.sh <csv>` | one set, in an interactive GPU session (`sxm2sh`, `voltash`, ...) instead of as a batch job |
| `env.sh` | sourced by the others | the queue default and the cache locations; not run directly |

So, once on the login node, then per run:

```bash
bash hpc/setup_env.sh                                    # once: environment + ESMFold weights

bsub < hpc/train.sh                                      # your training command
bsub < hpc/eval.sh                                       # every set that exists, controls included
bjobs                                                    # follow them; logs in hpc/logs/

bash hpc/run_eval.sh results/sequences/mock_data.csv      # or one set, interactively
```

Training needs no setup beyond `uv sync`; it is folding that needs `setup_env.sh`. A set name in
`hpc/eval.sh` with no sample CSV is skipped with a message rather than failing the job, so it can
be submitted before every model has been sampled. Both routes do the same work per set: the
metrics, ESMFold, and figures of the first five structures.

Queue, walltime and memory are the `#BSUB` lines at the top of each script rather than
command-line options. Run `bqueues` to see which GPU queues exist; every Nvidia queue works,
because torch on Linux is the CUDA 12.6 build (the PyPI build uses CUDA 13, which has no kernels
for V100s). Each job log starts with the GPU it got.

`hpc/env.sh` keeps the ESMFold weights and uv's package cache in `/work3/$USER` if it exists and in
`.cache/` inside the repo otherwise - never in the small home quota. If you run `uv` by hand
outside these scripts, `source hpc/env.sh` first, otherwise uv fills `~/.cache/uv` again.

**The sequence metrics need no GPU**, so only folding has to come here at all:

```bash
uv run python eval/eval.py --samples results/sequences/my_samples.csv   # runs on your laptop
```

## Layout

```
data/              train.csv, val.csv, test.csv - the frozen split, `sequence,family`
src/protgen/
  data/            ProteinDataset, the vocabulary and the family ids
  models/          empty: your architectures
  metrics/         the measurements eval/ is built from
    sequences.py   reading sample CSVs, validity, identity, k-mer divergence, the summary table
    controls.py    the real and shuffled control sets
    fold.py        ESMFold (needs a GPU and the eval group)
    structures.py  secondary structure, compactness, TM-score
eval/              the entry points, runnable anywhere: eval.py, make_controls.py,
                   compare_structures.py, plot_structure.py. What the numbers mean is in
                   eval/README.md
hpc/               how to run them on the cluster: the #BSUB job scripts, the setup script and the
                   shared settings. logs/ is where a job's .out/.err land
notebooks/         introduction.ipynb - the dataset class in a few cells
config/            empty: your Hydra config, if you use one
results/
  sequences/       sample CSVs (`sequence` column). mock_data.csv is checked in; the rest is
                   gitignored, as is results/eval/
```
