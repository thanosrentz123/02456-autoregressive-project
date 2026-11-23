from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset
from protgen.models.train_CG import load_transformer

PHASE1 = Path("results/checkpoints/1_2_Transformer_model_trained.pth")
MODEL_CODE = Path("src/protgen/models/1_2_Transformer_train.py")
CHECKPOINT = Path("results/phase3/checkpoints/judge_best.pt")


def prefix_batch(tokens, lengths):
    x = tokens[:, :128].clone()
    positions = torch.arange(128, device=tokens.device)[None, :]
    x[positions > lengths[:, None]] = 0
    return x


class JudgeClassifier(nn.Module):

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False
        self.head = nn.Linear(128, 3)

    def train(self, mode=True):
        super().train(mode)
        self.transformer.eval()
        return self

    def forward(self, tokens, lengths):
        with torch.no_grad():
            positions = torch.arange(tokens.size(1), device=tokens.device)
            h = self.transformer.seq_embedding(tokens) + self.transformer.pos_embedding(positions)
            padding = tokens.eq(0)
            h = self.transformer.dol1(h, padding)
            h = self.transformer.dol2(h, padding)
        keep = torch.arange(tokens.size(1), device=tokens.device)[None, :] <= lengths[:, None]
        pooled = (h * keep.unsqueeze(-1)).sum(1) / keep.sum(1, keepdim=True)
        return self.head(pooled)


def main():
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    transformer = load_transformer(MODEL_CODE.resolve(), PHASE1, device)
    model = JudgeClassifier(transformer).to(device)

    train_loader = DataLoader(ProteinDataset("data/train.csv"), batch_size=32, shuffle=True)
    val_loader = DataLoader(ProteinDataset("data/val.csv"), batch_size=32)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0

    for epoch in range(1, 16):
        model.train()
        for x, _, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(prefix_batch(x, lengths), lengths), y).backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, _, y, lengths in val_loader:
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                correct += (model(prefix_batch(x, lengths), lengths).argmax(-1) == y).sum().item()
                total += len(y)
        accuracy = correct / total
        flag = ""
        if accuracy > best:
            best, flag = accuracy, "  <- saved"
            torch.save(model.head.state_dict(), CHECKPOINT)
        print(f"epoch {epoch:2d} | val accuracy {accuracy:.3f}{flag}", flush=True)


if __name__ == "__main__":
    main()