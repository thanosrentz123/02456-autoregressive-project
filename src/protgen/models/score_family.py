import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from protgen.data import AA, FAMILIES
from protgen.models.classifier_guidance import ProteinClassifier
from protgen.models.judge import JudgeClassifier
from protgen.models.train_CG import load_transformer

PHASE1 = Path("results/checkpoints/1_2_Transformer_model_trained.pth")
MODEL_CODE = Path("src/protgen/models/1_2_Transformer_train.py")
HEADS = {"guide": Path("results/checkpoints/classifier_guidance_best.pt"),"judge": Path("results/phase3/checkpoints/judge_best.pt")}
SEQUENCES = Path("results/phase3/sequences")
OUT = Path("results/phase3/eval")
VOCAB = {residue: i + 3 for i, residue in enumerate(AA)}


def encode(sequences, device):

    tokens = torch.zeros(len(sequences), 128, dtype=torch.long)
    lengths = torch.zeros(len(sequences), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        residues = [VOCAB[c] for c in str(sequence).upper() if c in VOCAB][:127]
        tokens[i, 0] = 1                                   # <BOS>
        tokens[i, 1:len(residues) + 1] = torch.tensor(residues, dtype=torch.long)
        lengths[i] = len(residues)
    return tokens.to(device), lengths.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictor", choices=["guide", "judge"], default="guide")
    p.add_argument("--target", default="AcyP", choices=list(FAMILIES))
    p.add_argument("--sets", nargs="*", help="sample CSVs; default: every cg_/cfg_/control_ set")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = load_transformer(MODEL_CODE.resolve(), PHASE1, device)
    build = ProteinClassifier if args.predictor == "guide" else JudgeClassifier
    model = build(transformer).to(device)
    model.head.load_state_dict(
        torch.load(HEADS[args.predictor], map_location=device, weights_only=True))
    model.eval()

    paths = [Path(s) for s in args.sets] if args.sets else sorted(
        path for pattern in ("cg_*.csv", "cfg_*.csv", "control_*.csv")
        for path in SEQUENCES.glob(pattern))
    target = FAMILIES.index(args.target)

    rows = []
    for path in paths:
        sequences = pd.read_csv(path)["sequence"].tolist()
        tokens, lengths = encode(sequences, device)
        with torch.no_grad():
            probs = F.softmax(model(tokens, lengths), dim=-1)
        rows.append({"set": path.stem, "n": len(sequences), "target": args.target,
                     "frac_target": (probs.argmax(-1) == target).float().mean().item(),
                     "mean_p_target": probs[:, target].mean().item()})
        print(f"{path.stem:22s} n={rows[-1]['n']:4d} "
              f"frac={rows[-1]['frac_target']:.3f} mean_p={rows[-1]['mean_p_target']:.3f}")

    out = OUT / f"family_membership_{args.predictor}_{args.target}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()