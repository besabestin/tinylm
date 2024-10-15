import torch
import torch.nn as nn
import torch.nn.functional as F

from loader import Corpus, Dictionary

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
fn = './dataset/children_stories.train'
fn = '../babylm_data/babylm_100M/switchboard.train'
fn = '../babylm_data/babylm_100M/children_stories.train'

fnames = [
    '../babylm_data/babylm_100M/gutenberg.train',
    '../babylm_data/babylm_100M/qed.train',
    '../babylm_data/babylm_100M/switchboard.train',
    '../babylm_data/babylm_100M/children_stories.train'
]

context_length = 32
batch_size = 256
corpus = Corpus(fnames, batch_size=batch_size, context_size=context_length)
print(len(corpus.dictionary))

vocab_size = len(corpus.dictionary)
ndim = 48
niter = 50000
# learning_rate = 1e-04
learning_rate = 1e-03
learning_rate = 5e-04
ndecoders = 6

nheads = 4
pdrop = 0.1

train_model = False

class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pos = torch.arange(context_length).unsqueeze(-1)
        div_term = 1/(10000**(torch.arange(0, ndim, 2)/ndim))
        inner_term = pos * div_term
        pe = torch.zeros(context_length, ndim)
        pe[:, ::2] = torch.sin(inner_term)
        pe[:, 1::2] = torch.cos(inner_term)
        self.register_buffer('pos_encoding', pe)


    def forward(self, x):
        x = x + self.pos_encoding
        return x



class AttentionHead(nn.Module):
    def __init__(self, nheads) -> None:
        super().__init__()
        head_dim = ndim // nheads
        self.query_proj = nn.Linear(ndim, head_dim, bias=False)
        self.key_proj = nn.Linear(ndim, head_dim, bias=False)
        self.value_proj = nn.Linear(ndim, head_dim, bias=False)
        self.final_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        wei = q@k.transpose(-2, -1)
        wei = wei.masked_fill(self.tril == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = wei@v
        return self.final_proj(wei)



class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, nheads) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(nheads) for _ in range(nheads)])
        self.dropout = nn.Dropout(p=pdrop)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(out)


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.masked_attention = MaskedMultiHeadAttention(nheads)
        self.layernorm = nn.LayerNorm(ndim)

    def forward(self, x):
        x = x + self.masked_attention(x)
        x = self.layernorm(x)
        return x


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(ndim, ndim * 4),
            nn.ReLU(),
            nn.Linear(ndim * 4, ndim)
        )

    def forward(self, x):
        x = self.ffwd(x)
        return x


class LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, ndim)
        self.positional_encoding = PositionalEncoding()
        self.decoders = nn.Sequential(*[Decoder() for _ in range(ndecoders)])
        self.ffwd = FeedForward()
        self.layernorm = nn.LayerNorm(ndim)
        self.proj = nn.Linear(ndim, vocab_size)
        self.dropout1 = nn.Dropout(p=pdrop)
        self.dropout2 = nn.Dropout(p=pdrop)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.decoders(x)
        x = self.layernorm(x + self.ffwd(x))
        x = self.proj(x)
        x = F.log_softmax(x, dim=-1)
        return x


# tinishim yihun tlkm ybla, leserategn enklfu titaftewalech
# back to the fundamentals
# back to the fundamentals
@torch.no_grad()
def evaluate_loss():
    # let's do this for like 10 batches
    entire_loss = 0.
    eval_batches = 10
    for i in range(eval_batches):
        model.eval()
        X, y = corpus.get_batch('test')
        out = model(X)
        out = out.view(batch_size * context_length, vocab_size)
        y = y.view(batch_size * context_length, 1)
        loss = loss_fn(out, y.flatten())
        entire_loss += loss.item()
    
    return entire_loss / eval_batches



model = LanguageModel().to(device)
# the optimizer
optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_fn = nn.NLLLoss()

if train_model:
    for i in range(niter):
        model.train()
        X, y = corpus.get_batch('train')
        out = model(X)
        out = out.view(batch_size * context_length, vocab_size)
        y = y.view(batch_size*context_length, 1)
        loss = loss_fn(out, y.flatten())
        if i%500 == 0:
            val_loss = evaluate_loss()
            print(f"At iteration: {i + 1} | training loss: {loss.item()}, val loss: {val_loss}")
        loss.backward()
        optim.step()
        # clear gradient here.
        optim.zero_grad(set_to_none=True)

    torch.save(model.state_dict(), './lm.pt')
else:
    model.load_state_dict(torch.load('./lm.pt'))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'total number of parameters: {total_params}')

    with torch.no_grad():
        # prompt = "I am lucky to love you baby <eol> I am lucky to have known you <eol> I am lucky to have you I am lucky to have seen these days <eol> baby I will always"
        prompt = "I try to tell her , I try and tell her that she is smart and that she can try and have some trust in her but so far it hasn't really been"
        # prompt = "I should eat something only if I am hungry and I should eat something that is healthy and I should eat something that is good for me and I should eat something that is good for my body"
        tokens = corpus.dictionary.encode(prompt.split()[:context_length])
        generate_length = 100
        for _ in range(generate_length):
            _input = torch.tensor(tokens[len(tokens)-context_length:], dtype=torch.long)
            _input = _input.view(1,context_length).to(device)
            out = model(_input)
            B, T, C = out.shape # expecting B to be 1
            out = out.view(T, C)
            out = F.softmax(out, dim=-1)
            #print(f'out shape {out.shape}')
            prev_token = torch.multinomial(out[0,:], 1)
            #print(f'{corpus.dictionary.decode([prev_token.item()])}')
            nxt_token = torch.multinomial(out[-1,:], 1)
            tokens.append(nxt_token.item())
        print(corpus.dictionary.decode(tokens))
