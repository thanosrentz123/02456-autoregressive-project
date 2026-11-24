"""Classifier guidance for the group's frozen causal Transformer.

Place this file in src/protgen/models/classifier_guidance.py.
The original Transformer and TA's data code are NOT changed.
"""
import torch
from torch import nn
import torch.nn.functional as F

AA = "ACDEFGHIKLMNPQRSTVWY"
TOKEN_TO_AA = {i + 3: aa for i, aa in enumerate(AA)}



class ProteinClassifier(nn.Module):
    """Predict protein family from a partially observed sequence."""

    def __init__(self, transformer):
        super().__init__()

        self.transformer = transformer

        # Freeze pretrained Transformer parameters
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False

        self.transformer.eval()

        # Replace the original output layer with a classifier
        self.head = nn.Linear(transformer.ff1.out_features, 3)

    def train(self, mode=True):
        super().train(mode)
        self.transformer.eval()
        return self

    def forward(self, tokens, lengths):

        with torch.no_grad():

            positions = torch.arange(
                tokens.size(1), device=tokens.device
            )

            h = (
                self.transformer.seq_embedding(tokens)
                + self.transformer.pos_embedding(positions)
            )

            padding = tokens.eq(0)

            h = self.transformer.dol1(h, padding)
            h = self.transformer.dol2(h, padding)

            # Include pretrained hidden feedforward layer
            h = self.transformer.ff1(h)
            h = self.transformer.relu(h)

        # Extract the last observed token
        last = h[
            torch.arange(tokens.size(0), device=tokens.device),
            lengths
        ]

        return self.head(last)



@torch.no_grad()
def generate(transformer, classifier, family=0, guidance=1.0,
             n=10, max_length=127, temperature=1.0, device="cpu"):
    """Generate sequences using log p(next) + lambda * log p(family | prefix+next).

    Scores each of 20 amino acids plus EOS. For EOS, uses the classifier
    probability of the current prefix (no new residue); this is an approximation.
    """
    if not (0 <= family <= 2):
        raise ValueError("family must be 0=AcyP, 1=HPr or 2=HU")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not (1 <= max_length <= 127):
        raise ValueError("max_length must be between 1 and 127")
    transformer.eval()
    classifier.eval()
    sequences = []

    for _ in range(n):
        prefix = [1]  # BOS
        for step in range(max_length + 1):
            observed = len(prefix) - 1
            tokens = torch.zeros(1, 128, dtype=torch.long, device=device)
            tokens[0, :len(prefix)] = torch.tensor(prefix, device=device)
            lm_logits = transformer(tokens)[0, len(prefix) - 1]
            allowed = list(range(3, 23)) + [2]  # 20 residues and EOS
            if observed == 0:
                allowed.remove(2)  # no empty proteins
            if observed >= max_length:
                allowed = [2]

            # Each candidate residue is appended to the observed prefix.
            candidate_tokens = tokens.expand(len(allowed), -1).clone()
            candidate_lengths = torch.full((len(allowed),), observed, device=device, dtype=torch.long)
            for j, token in enumerate(allowed):
                if token != 2:
                    candidate_tokens[j, len(prefix)] = token
                    candidate_lengths[j] += 1
            family_log_probs = F.log_softmax(
                classifier(candidate_tokens, candidate_lengths), dim=-1
            )[:, family]
            scores = F.log_softmax(lm_logits, dim=-1)[allowed] + guidance * family_log_probs
            chosen = allowed[torch.distributions.Categorical(logits=scores / temperature).sample().item()]
            if chosen == 2:
                break
            prefix.append(chosen)
        sequences.append("".join(TOKEN_TO_AA[token] for token in prefix[1:]))
    return sequences
