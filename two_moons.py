# %%
from sklearn.datasets import make_moons
import torch
import tqdm
from matplotlib import pyplot as plt
from monotonenorm import direct_norm, GroupSort
from losses import DynamicHingeLoss

torch.manual_seed(1)

BATCHSIZE = -1
EPOCHS = 1000
WIDTH = 1024
LR = 1e-3
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
        # if i % (EPOCHS//100) == 0:
        #     with torch.no_grad():
        #         y_pred = model(x_val)
        #         val_loss = criterion(y_pred, y_val)
        #         val_acc = accuracy(y_pred, y_val)
        # pbar.set_description(f"Train: L: {loss.item():.3f} A: {acc:.1f} Val: L: {val_loss.item():.3f} A: {val_acc:.1f}")
        pbar.set_description(f"Train: L: {loss.item():.3f} A: {acc:.1f}")
    return model
# %%
def bce_temperature(tau=1):
    return lambda y_pred, y_true: torch.nn.BCEWithLogitsLoss()(y_pred * tau, y_true)
def mse():
    return lambda y_pred, y_true: torch.nn.MSELoss()(y_pred.sigmoid(), y_true)
def hinge(margin=0.1, x=None, y_true=None):
    return lambda y_pred, y_true: DynamicHingeLoss(margin=margin, p=1, x=x, y_true=y_true)(y_pred, y_true).log()

model_hinge1 = train("Lipschitz", hinge(1))
model_hinge = train("Lipschitz", hinge(0.1))
model_dhinge = train("Lipschitz", hinge(0.01, x_train, y_train))
model_mse = train("Lipschitz", mse())
model_bce = train("Lipschitz", bce_temperature(1))
model_bce_h = train("Lipschitz", bce_temperature(16))

# %%
# Model heatmap
x = torch.linspace(-2, 2, 100)
y = torch.linspace(-2, 2, 100)
X, Y = torch.meshgrid(x, y)
def plot_model(model, ax):
    Z = model(torch.stack([X, Y], dim=-1)).sigmoid_().detach().numpy()
    Z = Z.reshape(X.shape)
    mp = ax.contourf(X, Y, Z, levels=100, cmap="RdBu", vmin=0, vmax=1)
    ax.plot(x_train[y_train.squeeze() == 0, 0], x_train[y_train.squeeze() == 0, 1], "r.")
    ax.plot(x_train[y_train.squeeze() == 1, 0], x_train[y_train.squeeze() == 1, 1], "b.")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    return mp

# %%
fig, axes = plt.subplots(1, 6, figsize=(20, 4))
models = [model_mse, model_bce, model_bce_h, model_hinge, model_hinge1, model_dhinge]
titles = ["MSE", "BCE", "BCE high temp", "Hinge 0.1", "Hinge 1", "Dynamic Hinge"]
axes = axes.flatten()
for ax, model in zip(axes, models):
    mp = plot_model(model, ax)
    ax.set_title(titles.pop(0))
    with torch.no_grad():
        y_pred = model(x_val)
        val_acc = ((y_pred > 0).long() == y_val).sum().item() / y_pred.shape[0] * 100
    ax.text(0.96, 0.96, f"Acc: {val_acc:.1f}%", horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    plt.colorbar(mp, ax=ax)
# %%
# y_pred = model_dhinge(x_train)
# loss = DynamicHingeLoss(margin=0.01, x=x_train, y_true=y_train)
# print(loss(y_pred, y_train))
# sign = 2 * y_train - 1
# print((loss.margin - sign * y_pred).relu().mean())
# print((loss.dynamic_margin + loss.margin - sign * y_pred).relu().mean())

# %%
