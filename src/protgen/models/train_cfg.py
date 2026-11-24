import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset
from protgen.models.cfg_transformer import (FAMILIES, NULL, ConditionalTransformer, sample, warm_start)

PHASE1 = "results/checkpoints/1_2_Transformer_model_trained.pth"
CHECKPOINT = Path("results/phase3/checkpoints/cfg_transformer_best.pt")
SEQUENCES = Path("results/phase3/sequences")
P_DROP = 0.15


def loss_on(model, x, y):
    """Phase 1's convention: predict x[:, 1:] from x[:, :-1], ignoring <PAD> targets."""
    logits = model(x[:, :-1], y)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           x[:, 1:].reshape(-1), ignore_index=0)


def train(model, device, epochs):
    train_loader = DataLoader(ProteinDataset("data/train.csv"), batch_size=32, shuffle=True)
    val_loader = DataLoader(ProteinDataset("data/val.csv"), batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, _, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            y = y.masked_fill(torch.rand(y.shape, device=device) < P_DROP, NULL)
            optimizer.zero_grad()
            loss = loss_on(model, x, y)
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        cond = uncond = 0.0
        with torch.no_grad():
            for x, _, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                cond += loss_on(model, x, y).item()
                uncond += loss_on(model, x, torch.full_like(y, NULL)).item()
        cond, uncond = cond / len(val_loader), uncond / len(val_loader)

        flag = ""
        if (cond + uncond) / 2 < best:
            best = (cond + uncond) / 2
            torch.save(model.state_dict(), CHECKPOINT)
            flag = "  <- saved"
        print(f"epoch {epoch:3d} | train {total / len(train_loader):.4f} | "
              f"val cond {cond:.4f} uncond {uncond:.4f} gap {uncond - cond:+.4f}{flag}",
              flush=True)


def generate(model, device, family, ws, n, seed):
    SEQUENCES.mkdir(parents=True, exist_ok=True)
    null = family == "null"
    runs = [(None, 0.0, "cfg_null")] if null else \
           [(FAMILIES.index(family), w, f"cfg_{family}_w{w:g}") for w in ws]

    for label, w, name in runs:
        torch.manual_seed(seed)
        seqs = sample(model, label, w, n, device)
        path = SEQUENCES / f"{name}.csv"
        pd.DataFrame({"sequence": seqs,
                      "family": "uncond" if null else family,
                      "setting": "uncond" if null else f"w={w:g}",
                      "lambda_equiv": 0.0 if null else 1 + w}).to_csv(path, index=False)
        print(f"{path}  n={len(seqs)}  mean length {sum(map(len, seqs)) / len(seqs):.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "sample"], default="train")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--family", default="AcyP", choices=list(FAMILIES) + ["null"])
    p.add_argument("--w", type=float, nargs="+", default=[0, 0.5, 1, 2, 3])
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ConditionalTransformer().to(device)

    if args.mode == "train":
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        warm_start(model, PHASE1, device)
        print(f"Warm started from {PHASE1};{sum(t.numel() for t in model.parameters()):,} parameters", flush=True)
        train(model, device, args.epochs)
    else:
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
        generate(model, device, args.family, args.w, args.n, args.seed)


if __name__ == "__main__":
    main()