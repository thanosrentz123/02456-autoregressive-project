"""How accurate are the two family predictors, and how much sequence do they need?

    uv run python eval/predictor_accuracy.py

Reports accuracy at 25 / 50 / 75 / 100 % of each validation protein. The partial
columns are what matters for classifier guidance: it calls the predictor on
half-finished sequences at every generation step.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset
from protgen.models.classifier_guidance import ProteinClassifier
from protgen.models.judge import JudgeClassifier
from protgen.models.train_CG import load_transformer, validate

PHASE1 = Path("results/checkpoints/1_2_Transformer_model_trained.pth")
MODEL_CODE = Path("src/protgen/models/1_2_Transformer_train.py")
PREDICTORS = [
    ("guide", ProteinClassifier, Path("results/checkpoints/classifier_guidance_best.pt")),
    ("judge", JudgeClassifier, Path("results/phase3/checkpoints/judge_best.pt")),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    transformer = load_transformer(MODEL_CODE.resolve(), PHASE1, device)
    loader = DataLoader(ProteinDataset("data/val.csv"), batch_size=32)

    print(f"\n{'predictor':10s} {'25%':>8s} {'50%':>8s} {'75%':>8s} {'100%':>8s}")
    for name, build, checkpoint in PREDICTORS:
        if not checkpoint.exists():
            print(f"{name:10s} missing {checkpoint}")
            continue
        model = build(transformer).to(device)
        model.head.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True))
        accuracy = validate(model, loader, device)
        print(f"{name:10s}" + "".join(f"{accuracy[f]:8.3f}" for f in (0.25, 0.5, 0.75, 1.0)),
              flush=True)


if __name__ == "__main__":
    main()
