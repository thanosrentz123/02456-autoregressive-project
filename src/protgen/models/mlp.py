"""

Predicts the next token from the `context_size` tokens before it. Written against
protgen.data.ProteinDataset: it takes `x` (B, 129) and targets `x_shifted`, so a
whole padded batch is scored in one forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from protgen.data import AA, IGNORE_INDEX, total_tokens

PAD_IDX = total_tokens.index("<PAD>")  # 0
BOS_IDX = total_tokens.index("<BOS>")  # 1
EOS_IDX = total_tokens.index("<EOS>")  # 2


class MLPLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size=len(total_tokens),   # 23
        context_size=3,
        embed_dim=16,
        hidden_dims=(256, 256),
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size

        # Lookup table: token id -> vector of size embed_dim.
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # Trunk and output layer are kept apart so a classification head can be
        # attached to the trunk in phase 2.
        layers = []
        in_dim = context_size * embed_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, vocab_size)
        self.hidden_size = in_dim

    def _windows(self, x):
    
        pad = x.new_full((x.size(0), self.context_size - 1), BOS_IDX)
        padded = torch.cat([pad, x], dim=1)              # (B, L + c - 1)
        return padded.unfold(dimension=1, size=self.context_size, step=1)

    def features(self, x):
        
        context = self._windows(x)                       # (B, L, c)
        embedded = self.embed(context)                   # (B, L, c, d)
        return self.trunk(embedded.flatten(start_dim=2)) # (B, L, H)

    def forward(self, x):
        """(B, L) token ids -> (B, L, vocab_size) next-token logits."""
        return self.head(self.features(x))

    def loss(self, x, x_shifted):
        """Mean cross-entropy over real positions; padding is IGNORE_INDEX (-1)."""
        logits = self(x)
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            x_shifted.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    @torch.no_grad()
    def sample(self, n, max_length=127, temperature=1.0, device="cpu"):
        """Generate n sequences as residue strings."""
        self.eval()
        context = torch.full((n, self.context_size), BOS_IDX,
                             dtype=torch.long, device=device)
        out = torch.empty((n, 0), dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)

        for _ in range(max_length):
            hidden = self.trunk(self.embed(context).flatten(start_dim=1))
            logits = self.head(hidden)           # (n, V)
            logits[:, PAD_IDX] = -float("inf")   # never emit <PAD>
            logits[:, BOS_IDX] = -float("inf")   # <BOS> only ever starts a sequence
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)

            nxt = torch.where(done, torch.full_like(nxt, PAD_IDX), nxt)
            done |= nxt == EOS_IDX

            out = torch.cat([out, nxt.unsqueeze(1)], dim=1)
            context = torch.cat([context[:, 1:], nxt.unsqueeze(1)], dim=1)
            if bool(done.all()):
                break

        residue = {total_tokens.index(a): a for a in AA}
        return ["".join(residue[int(t)] for t in row if int(t) in residue)
                for row in out]