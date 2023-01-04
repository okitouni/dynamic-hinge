# Goal: Train a model to classify between two gaussians
# Need to get a loss whose argmin is Optimal bayes classifier
# when using a LipNN

import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
from monotonenorm import direct_norm, GroupSort
from losses import DynamicHingeLoss

def accuracy(y_pred, y_true):
    return (torch.sign(y_pred) == y_true).sum().item() / y_pred.shape[0]

torch.manual_seed(0)
EPOCHS = 1000
npoints = 100
PLOT = True

dist1 = torch.distributions.Normal(0.5, 1)
dist2 = torch.distributions.Normal(-0.5, 0.9)

X = torch.cat((dist1.sample((npoints//2, 1)), dist2.sample((npoints//2, 1))))
Y = torch.cat((torch.ones(npoints//2, 1), -torch.ones(npoints//2, 1)))

X, argsort = X.sort(0)
Y = Y[argsort.view(-1)]

# likelihood ratio
llhood = torch.exp(dist1.log_prob(X) - dist2.log_prob(X))
# bayes optimal classifier
llhood = 2 * (llhood / (llhood + 1) )- 1
# best classifier accuracy

print("Optimal Bayes Accuracy:", accuracy(llhood, Y))
hidden_dim = 8

norm = lambda x, *args, **kwargs: x
# norm = lambda x, *args, **kwargs: direct_norm(x, *args, **kwargs)
model = nn.Sequential(
    norm(nn.Linear(1, hidden_dim), kind="one-inf"),
    GroupSort(hidden_dim//2),
    norm(nn.Linear(hidden_dim, hidden_dim), kind="inf"),
    GroupSort(hidden_dim//2),
    norm(nn.Linear(hidden_dim, 1), kind="inf"),
    nn.Sigmoid(),
)


criterion = lambda pred, target : nn.functional.binary_cross_entropy(pred, (target+1)/2)
# criterion = DynamicHingeLoss(margin=1, x=X, y_true=Y, reduction='mean') 
# criterion = nn.SoftMarginLoss()
# criterion =  

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

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
    xrange = X
    with torch.no_grad(): yrange = model(xrange)


    # likelihood ratio
    llhood = torch.exp(dist1.log_prob(xrange) - dist2.log_prob(xrange))
    # Optimal bayes classifier
    llhood = 2 * llhood / (llhood + 1) - 1
    
    plt.scatter(X, Y)
    plt.plot(xrange, llhood, label="OptimalBayes",)
    plt.plot(xrange, yrange, label="NN")
    # different preds
    acc_bayes = accuracy(llhood, Y)
    acc_model = accuracy(yrange, Y)
    print("Accuracy Optimal:", acc_bayes, "Accuracy model:", acc_model)
    # mask = (yrange > 0.5) != (llhood > 0.5)
    # print(mask.sum())
    # plt.scatter(xrange[mask], yrange[mask], label="different preds", c=Y[mask], marker=".")
    plt.legend()
    plt.savefig("data.png")