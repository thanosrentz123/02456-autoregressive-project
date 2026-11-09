"""Train theMLP language model. uv run python train.py
"""

import math
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset
from protgen.metrics import write_sequences
from protgen.models.mlp import MLPLanguageModel

CONFIG = dict(context_size=3, embed_dim=16, hidden_dims=(256, 256), dropout=0.1)
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
PATIENCE = 5          # epochs without val improvement before stopping
N_SAMPLES = 256       # match eval/make_controls.py --num
SEED = 0


def device():
    if torch.backends.mps.is_available():   # Apple silicon
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, dev):
    """Mean cross-entropy per real token over a whole split."""
    model.eval()
    total, count = 0.0, 0
    for x, x_shifted, _, _ in loader:
        x, x_shifted = x.to(dev), x_shifted.to(dev)
        n = int((x_shifted != -1).sum())
        total += float(model.loss(x, x_shifted)) * n
        count += n
    return total / count


def unigram_perplexity(dataset):
    """Baseline: predict the next token from residue frequencies alone."""
    counts = Counter()
    for sequence in dataset.df["sequence"]:
        counts.update(sequence)
        counts["<EOS>"] += 1
    n = sum(counts.values())
    return math.exp(-sum(c / n * math.log(c / n) for c in counts.values()))


def main():
    torch.manual_seed(SEED)
    dev = device()
    print(f"device: {dev}")

    train_data = ProteinDataset("data/train.csv")
    val_data = ProteinDataset("data/val.csv")
    print(train_data)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = MLPLanguageModel(**CONFIG).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"unigram baseline perplexity: {unigram_perplexity(train_data):.3f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history, best, bad_epochs = [], float("inf"), 0
    Path("runs").mkdir(exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total, count = 0.0, 0
        for x, x_shifted, _, _ in train_loader:
            x, x_shifted = x.to(dev), x_shifted.to(dev)

            loss = model.loss(x, x_shifted)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = int((x_shifted != -1).sum())
            total += loss.item() * n
            count += n

        train_loss = total / count
        val_loss = evaluate(model, val_loader, dev)
        history.append((epoch, train_loss, val_loss))
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"val ppl {math.exp(val_loss):.3f}")

        if val_loss < best - 1e-4:
            best, bad_epochs = val_loss, 0
            torch.save({"state_dict": model.state_dict(), "config": CONFIG},
                       "runs/mlp.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stop at epoch {epoch}")
                break

    print(f"\nbest val loss {best:.4f}  (perplexity {math.exp(best):.3f})")

    with open("runs/mlp_history.csv", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for row in history:
            f.write("{},{:.6f},{:.6f}\n".format(*row))

    # Sample from the best checkpoint, for eval/eval.py.
    model.load_state_dict(torch.load("runs/mlp.pt")["state_dict"])
    sequences = model.sample(N_SAMPLES, device=dev)
    path = write_sequences("results/sequences/mlp.csv",
                           [{"sequence": s} for s in sequences])
    lengths = [len(s) for s in sequences]
    print(f"wrote {len(sequences)} sequences to {path}  "
          f"(length min {min(lengths)}, mean {sum(lengths)/len(lengths):.1f}, "
          f"max {max(lengths)})")


if __name__ == "__main__":
    main()