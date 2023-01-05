# Goal: Train a model to classify between two gaussians
# Need to get a loss whose argmin is Optimal bayes classifier
# when using a LipNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
from monotonenorm import direct_norm, GroupSort
from losses import DynamicHingeLoss, HingeLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(y_pred, y_true):
    return ((y_pred > y_pred.quantile(.5)) == y_true).sum().item() / y_pred.shape[0]

torch.manual_seed(0)
EPOCHS = 10000
npoints = 5000
PLOT = True

dist1 = torch.distributions.Normal(-1, 1)
dist2 = torch.distributions.Normal(1, 1)

sample_shape = torch.Size([npoints//2, 1])

X = torch.cat((dist1.sample(sample_shape), dist2.sample(sample_shape))).to(device)
Y = torch.cat((torch.ones(npoints//2, 1), torch.zeros(npoints//2, 1))).to(device)

X, argsort = X.sort(0)
Y = Y[argsort.view(-1)]

# likelihood ratio
llhood = torch.exp(dist1.log_prob(X) - dist2.log_prob(X))
# bayes optimal classifier
llhood = llhood / (llhood + 1)
# best classifier accuracy

print("Optimal Bayes Accuracy:", accuracy(llhood, Y))

class Multiply(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
    def forward(self, x):
        return x * self.factor

hidden_dim = 256
multiplier = 10

# norm = lambda x, *args, **kwargs: x
norm = direct_norm
model = nn.Sequential(
    norm(nn.Linear(1, hidden_dim), kind="one-inf"),
    GroupSort(hidden_dim//2),
    norm(nn.Linear(hidden_dim, hidden_dim), kind="inf"),
    GroupSort(hidden_dim//2),
    norm(nn.Linear(hidden_dim, hidden_dim), kind="inf"),
    GroupSort(hidden_dim//2),
    norm(nn.Linear(hidden_dim, hidden_dim), kind="inf"),
    GroupSort(hidden_dim//2),
    norm(nn.Linear(hidden_dim, hidden_dim), kind="inf"),
    GroupSort(hidden_dim//2),
    norm(nn.Linear(hidden_dim, 1), kind="inf"),
    Multiply(multiplier),
#    nn.Sigmoid(),
).to(device)


# criterion = F.binary_cross_entropy
criterion = DynamicHingeLoss(margin=.1, x=X, y_true=Y, reduction='mean', scale=True)
# criterion = HingeLoss(margin=10, scale=True)

lr = 1e-1 if norm == direct_norm else 1e-2
lr *= 1/multiplier

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)

pbar = tqdm.tqdm(range(EPOCHS))
for i in pbar:
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
      acc = accuracy(y_pred, Y)
    pbar.set_description(f"Loss: {loss.item():.2f}, acc: {acc:.2f}")


if PLOT:
    X = X.cpu()
    Y = Y.cpu()
    model = model.cpu()
    if hasattr(criterion, "cpu"):
      criterion = criterion.cpu()
    xrange = X
    with torch.no_grad():
      yrange = model(xrange)
      # likelihood ratio
      llhood = torch.exp(dist1.log_prob(xrange) - dist2.log_prob(xrange))
      # Optimal bayes classifier
      llhood = llhood / (llhood + 1)

      if hasattr(criterion, "scale") and criterion.scale:
        yrange = (yrange + 1) / 2

      plt.scatter(X, Y)
      plt.plot(xrange, llhood, label="OptimalBayes",)
      plt.plot(xrange, yrange, label="NN")
      # different preds
      acc_bayes = accuracy(llhood, Y)
      acc_model = accuracy(yrange, Y)
      loss_bayes = criterion(llhood, Y).item()
      loss_model = criterion(yrange, Y).item()
      print(f"Bayes acc: {acc_bayes:.3f}, NN acc: {acc_model:.3f}, Bayes loss: {loss_bayes:.3f}, NN loss: {loss_model:.3f}")
      # cut value
      cut = yrange.quantile(.5)
      plt.plot(xrange, torch.ones_like(xrange) * cut, label="cut")
      plt.legend()
      plt.savefig("data.png")
