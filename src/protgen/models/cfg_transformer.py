import torch
from torch import nn
import torch.nn.functional as F
from protgen.data import AA, FAMILIES

PAD, BOS, EOS = 0, 1, 2
VOCAB_SIZE = 23
SEQ_LEN = 128
MAX_RESIDUES = 127
NULL = 3


class decoder_only_layer(nn.TransformerEncoderLayer):

    def __init__(self):
        super().__init__(d_model=128, nhead=4, batch_first=True, norm_first=True)
        self.register_buffer(
            "mask", torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1))

    def forward(self, x, padding_mask):
        return super().forward(x, src_mask=self.mask,
                               src_key_padding_mask=padding_mask, is_causal=True)


class ConditionalTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.seq_embedding = nn.Embedding(VOCAB_SIZE, 128)
        self.pos_embedding = nn.Embedding(SEQ_LEN, 128)
        self.label_embedding = nn.Embedding(4, 128)
        nn.init.zeros_(self.label_embedding.weight)
        self.dol1 = decoder_only_layer()
        self.dol2 = decoder_only_layer()
        self.ff1 = nn.Linear(128, 200)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(200, VOCAB_SIZE)

    def forward(self, x, y):
        padding_mask = x.eq(PAD)
        positions = torch.arange(x.size(1), device=x.device)
        h = (self.seq_embedding(x) + self.pos_embedding(positions)
             + self.label_embedding(y).unsqueeze(1))   # broadcast the label over all positions
        h = self.dol2(self.dol1(h, padding_mask), padding_mask)
        return self.ff2(self.relu(self.ff1(h)))


def warm_start(model, path, device):
    missing, unexpected = model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True), strict=False)
    assert set(missing) == {"label_embedding.weight"} and not unexpected, (missing, unexpected)


@torch.no_grad()
def sample(model, family, w, n, device):
    model.eval()
    labels = torch.cat([torch.full((n,), NULL if family is None else family, device=device),
                        torch.full((n,), NULL, device=device)])
    tokens = torch.zeros(n, SEQ_LEN, dtype=torch.long, device=device)
    tokens[:, 0] = BOS
    lengths = torch.zeros(n, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)

    for step in range(MAX_RESIDUES + 1):
        log_p = F.log_softmax(model(tokens.repeat(2, 1), labels)[:, step, :], dim=-1)
        scores = (1 + w) * log_p[:n] - w * log_p[n:]
        scores[:, [PAD, BOS]] = -torch.inf
        if step == 0:
            scores[:, EOS] = -torch.inf
        if step == MAX_RESIDUES:
            scores[:] = -torch.inf
            scores[:, EOS] = 0.0

        nxt = torch.multinomial(F.softmax(scores, dim=-1), 1).squeeze(1)
        write = ~done & (nxt != EOS)
        if step < MAX_RESIDUES:
            tokens[write, step + 1] = nxt[write]
            lengths[write] += 1
        done |= nxt == EOS
        if bool(done.all()):
            break

    return ["".join(AA[t - 3] for t in tokens[i, 1:1 + int(lengths[i])].tolist())
            for i in range(n)]