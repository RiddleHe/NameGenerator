import torch
import torch.nn.functional as F

import model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

words = open('names.txt', 'r').read().splitlines()

# build vocabulary
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."]=0
itos = {i:s for s,i in stoi.items()}

# shuffle the data
import random
random.seed(42)
random.shuffle(words)

# build dataset

context_size = 8

def build_dataset(files):
  X,Y = [], []
  for file in files:
    context = [0]*context_size
    for ch in file+".":
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix]
  X = torch.tensor(X, device=device)
  Y = torch.tensor(Y, device=device)
  return X,Y

x1 = int(len(words)*0.8)
x2 = int(len(words)*0.9)


Xtr, Ytr = build_dataset(words[:x1])
Xval, Yval = build_dataset(words[x1:x2])
Xtest, Ytest = build_dataset(words[x2:])

voc_size = 27
n_emb = 10
n_hidden = 60

model = Sequential(
    [Embedding(voc_size, n_emb),
     Flatten(2), Linear(n_emb*2, n_hidden), BatchNorm1d(n_hidden), Tanh(),
     Flatten(2), Linear(n_hidden*2, n_hidden), BatchNorm1d(n_hidden), Tanh(),
     Flatten(2), Linear(n_hidden*2, n_hidden), BatchNorm1d(n_hidden), Tanh(),
     Linear(n_hidden, voc_size), ]
)

with torch.no_grad():
  model.layers[-1].weights *= 0.1

for p in model.parameters():
  p.requires_grad=True

# training

max_steps = 200000
lossi = []

for i in range(max_steps):

  # mini-batch
  ix = torch.randint(0, Xtr.shape[0], (32,))
  Xb, Yb = Xtr[ix], Ytr[ix]

  # forward pass
  x = Xb
  logits = model(x)
  loss = F.cross_entropy(logits, Yb)

  # backward pass
  for p in model.parameters():
    p.grad = None
  loss.backward()

  # update
  lr = 0.1 if i<150000 else 0.05
  for p in model.parameters():
    p.data += - lr * p.grad

  # track stats
  if i%10000==0:
    print(f'{i}th iteration: loss {loss.item()}')
  lossi.append(loss.item())

plt.plot(torch.tensor(lossi).view(-1, 10000).mean(1))

for layer in model.layers:
  if isinstance(layer, BatchNorm1d):
    layer.training = False

# sample from the model

for i in range(20):
  out = []
  context = [0]*context_size
  while True:
    logits = model(torch.tensor([context]))
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1).item()

    if ix==0:
      break

    out.append(ix)
    context = context[1:] + [ix]

  print("".join(itos[x] for x in out))