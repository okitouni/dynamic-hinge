# %%
from sklearn.datasets import make_moons
import torch
import tqdm
from matplotlib import pyplot as plt
from monotonenorm import direct_norm, GroupSort

torch.manual_seed(1)

BATCHSIZE = -1
EPOCHS = 1000
WIDTH = 1024
LR = 1e-4
N_SAMPLES = 1000
x, y = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=1)
x -= x.mean(axis=0)
x /= x.std(axis=0)

x_train = torch.tensor(x[:-N_SAMPLES//5], dtype=torch.float32)
y_train = torch.tensor(y[:-N_SAMPLES//5], dtype=torch.float32).unsqueeze(1)
x_val = torch.tensor(x[-N_SAMPLES//5:], dtype=torch.float32)
y_val = torch.tensor(y[-N_SAMPLES//5:], dtype=torch.float32).unsqueeze(1)
# %%
def train(model_type, criterion):
    norm = direct_norm if model_type == "Lipschitz" else lambda x, *args, **kwargs: x
    act = GroupSort(WIDTH // 2) if model_type == "Lipschitz" else torch.nn.SiLU()
    model = torch.nn.Sequential(
        norm(torch.nn.Linear(2, WIDTH), kind="one-inf"),
        act,
        norm(torch.nn.Linear(WIDTH, WIDTH), kind="inf"),
        act,
        norm(torch.nn.Linear(WIDTH, WIDTH), kind="inf"),
        act,
        norm(torch.nn.Linear(WIDTH, 1), kind="inf"),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    accuracy = lambda y_pred, y_true: ((y_pred > 0).long() == y_true).sum().item() / y_pred.shape[0] * 100
    pbar = tqdm.tqdm(range(EPOCHS))
    for i in pbar:
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        acc = accuracy(y_pred, y_train)
        if i % (EPOCHS//100) == 0:
            with torch.no_grad():
                y_pred = model(x_val)
                val_loss = criterion(y_pred, y_val)
                val_acc = accuracy(y_pred, y_val)
        pbar.set_description(f"Train: L: {loss.item():.3f} A: {acc:.1f} Val: L: {val_loss.item():.3f} A: {val_acc:.1f}")
    return model
# %%
train("Lipschitz", torch.nn.BCEWithLogitsLoss())
# %%
# Model heatmap
x = torch.linspace(-2, 2, 100)
y = torch.linspace(-2, 2, 100)
X, Y = torch.meshgrid(x, y)
def plot_model(model, ax):
    Z = model(torch.stack([X, Y], dim=-1)).sigmoid_().detach().numpy()
    Z = Z.reshape(X.shape)
    mp = ax.contourf(X, Y, Z, levels=100, cmap="RdBu")
    ax.plot(x_train[y_train.squeeze() == 0, 0], x_train[y_train.squeeze() == 0, 1], "r.")
    ax.plot(x_train[y_train.squeeze() == 1, 0], x_train[y_train.squeeze() == 1, 1], "b.")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    return mp

fig, ax = plt.subplots()
mp = plot_model(model, ax)
# add a colorbar
plt.colorbar(mp)
# %%
