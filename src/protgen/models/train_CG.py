"""Train and sample a classifier on the group's frozen Transformer.

Run from the repository root. See --help for options.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# Run from the repository root (where data/ and results/ are located).
ROOT = Path.cwd()
from protgen.models.classifier_guidance import ProteinClassifier, generate


def load_transformer(model_code, checkpoint, device):
    """Import teammate's exact Transformer class without running its main()."""
    # Its import 'from make_data import *' needs the directory containing make_data.py.
    for folder in [ROOT / 'data', ROOT / 'scripts', ROOT,
                   ROOT / 'src' / 'protgen' / 'data', model_code.parent]:
        if (folder / 'make_data.py').exists():
            sys.path.insert(0, str(folder))
            break
    spec = importlib.util.spec_from_file_location('team_transformer', model_code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Transformer_model(d_model=128, nhead=4, d_ff=200, device=device).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def partial_batch(tokens, lengths, fraction=None):
    """Turn full proteins into partial prefixes, keeping BOS and masking the rest."""
    if fraction is None:
        # Random lengths; 25% of examples are full-length to learn clean prediction.
        chosen = (torch.rand(len(lengths), device=lengths.device) * lengths.float()).long() + 1
        full = torch.rand(len(lengths)) < 0.25
        chosen = torch.where(full.to(lengths.device), lengths, chosen)
    else:
        chosen = (lengths.float() * fraction).ceil().long().clamp(min=1)
    chosen = chosen.to(tokens.device)
    positions = torch.arange(128, device=tokens.device)[None, :]
    prefix = tokens[:, :128].clone()
    prefix[positions > chosen[:, None]] = 0  # BOS is position 0
    return prefix, chosen


def validate(model, loader, device):
    model.eval()
    result = {}
    with torch.no_grad():
        for fraction in [0.25, 0.50, 0.75, 1.0]:
            correct = total = 0
            for x, _, labels, lengths in loader:
                x, labels, lengths = x.to(device), labels.to(device), lengths.to(device)
                partial, observed = partial_batch(x, lengths, fraction)
                prediction = model(partial, observed).argmax(dim=-1)
                correct += (prediction == labels).sum().item()
                total += len(labels)
            result[fraction] = correct / total
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'sample'], default='train')
    parser.add_argument('--model-code', default='src/protgen/models/1_2_Transformer_train.py')
    parser.add_argument('--checkpoint', default='results/checkpoints/1_2_Transformer_model_trained.pth')
    parser.add_argument('--classifier-checkpoint', default='results/checkpoints/classifier_guidance_best.pt')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--family', choices=['AcyP', 'HPr', 'HU'], default='AcyP')
    parser.add_argument('--guidance', type=float, default=1.0)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    transformer = load_transformer(Path(args.model_code).resolve(), Path(args.checkpoint), device)
    classifier = ProteinClassifier(transformer).to(device)
    checkpoint_path = Path(args.classifier_checkpoint)

    if args.mode == 'sample':
        classifier.head.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        family_id = {'AcyP': 0, 'HPr': 1, 'HU': 2}[args.family]
        sequences = generate(transformer, classifier, family=family_id,
                             guidance=args.guidance, n=args.n,
                             temperature=args.temperature, device=device)
        out = Path('results/sequences') / f'cg_{args.family}_lambda{args.guidance:g}.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'sequence': sequences, 'family': args.family,
                      'setting': f'lambda={args.guidance:g}'}).to_csv(out, index=False)
        print('Saved:', out)
        return

    # Use TA's installed dataset class; do not replace or edit it.
    from protgen.data import ProteinDataset
    train_data = ProteinDataset('data/train.csv')
    val_data = ProteinDataset('data/val.csv')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    optimizer = torch.optim.Adam(classifier.head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best = -1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0
        for x, _, labels, lengths in train_loader:
            x, labels, lengths = x.to(device), labels.to(device), lengths.to(device)
            partial, observed = partial_batch(x, lengths)
            optimizer.zero_grad()
            loss = criterion(classifier(partial, observed), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        accuracy = validate(classifier, val_loader, device)
        print(f'Epoch {epoch:02d} | loss={total_loss / len(train_data):.4f} | '
              + ' '.join(f'{int(f*100)}%={a:.1%}' for f, a in accuracy.items()))
        # Use complete-sequence validation accuracy to select best checkpoint.
        if accuracy[1.0] > best:
            best = accuracy[1.0]
            torch.save(classifier.head.state_dict(), checkpoint_path)
            print('  Saved best classifier:', checkpoint_path)


if __name__ == '__main__':
    main()
