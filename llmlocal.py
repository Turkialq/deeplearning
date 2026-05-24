import torch
import torch.nn as nn
from torch.nn import functional as F
# Hyper-parameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32


with open("input.txt", encoding="utf-8") as f:
    text = f.read()


# Unuque characters in our dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)



# encode and decode popular ones (sentencePiece)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print (encode("hii there"))
print(decode(encode("hii there")))



# encode our dataset
data = torch.tensor(encode(text), dtype = torch.long)
print (data.shape, data.dtype)
print (data[:100])



n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]




torch.manual_seed(1337)
# data loader
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

print("----")

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print (f"when input is {context.tolist()} the target: {target}")



class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward (self, idx, targets = None):
        B, T  = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embeddings + position_embeddings
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate (self, idx, max_new_tokens):
        for _ in range (max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits [:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next), dim=1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
logits, loss = m(xb,yb)
print(logits.shape)
print(loss)        



optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for steps in range (100000):
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print("Epoch:", steps)

print(loss.item())


context = torch.zeros((1, 1), dtype=torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))



B, T, C = 4,8,32
x = torch.randn(B,T,C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)
q = query(x)
v = value(x)

wei  =  q @ k.transpose(-2, -1)

tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril == 0 , float('-inf'))
wei = F.softmax(wei, dim = -1)
out = wei @ v

print(wei[0])
out.shape