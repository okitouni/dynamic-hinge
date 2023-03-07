# %%
from sklearn.datasets import make_moons
import torch
import tqdm
from matplotlib import pyplot as plt
from monotonenorm import direct_norm, GroupSort
from losses import DynamicHingeLoss
import os

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

def get_model(model_type):
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
    return model
# %%
def train(model_type, criterion):
    model = get_model(model_type)
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

names =  ["MSE", "BCE", "BCEht", "Hinge1", "Hinge0.1", "DynamicHinge"]
loss_fns = [hinge(1), hinge(0.1), hinge(0.01, x_train, y_train), mse(), bce_temperature(1), bce_temperature(16)]
if not os.path.exists("models"):
    os.mkdir("models")
for name, loss_fn in zip(names, loss_fns):
    if os.path.exists(f"models/{name}.pt"):
        continue
    model = train("Lipschitz", loss_fn)
    torch.save(model.state_dict(), f"models/{name}.pt")

models = {}
for name in names:
    model = get_model("Lipschitz")
    model.load_state_dict(torch.load(f"models/{name}.pt"))
    model.eval()
    models[name] = model
# %%
# Model heatmap
plt.style.use("mystyle-bright.mplstyle")
x = torch.linspace(-2, 2, 100)
y = torch.linspace(-2, 2, 100)
X, Y = torch.meshgrid(x, y)
def plot_model(model, ax):
    Z = model(torch.stack([X, Y], dim=-1)).sigmoid_().detach().numpy()
    Z = Z.reshape(X.shape)
    Z = 2 * Z - 1
    mp = ax.contourf(X, Y, Z, levels=100, cmap="RdBu", vmin=-1, vmax=1)
    ax.plot(x_train[y_train.squeeze() == 0, 0], x_train[y_train.squeeze() == 0, 1], "r.")
    ax.plot(x_train[y_train.squeeze() == 1, 0], x_train[y_train.squeeze() == 1, 1], "b.")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    return mp

# %%
from matplotlib.ticker import FormatStrFormatter
one_bar = True
fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
titles = ["MSE","BCE", r"BCE ($\tau=1$)", r"Hinge ($m=1$)", r"Hinge ($m=0.1$)", r"Dynamic Hinge"]
axes = axes.flatten()
for ax, model in zip(axes, models.values()):
    mp = plot_model(model, ax)
    ax.set_title(titles.pop(0))
    with torch.no_grad():
        y_pred = model(x_val)
        val_acc = ((y_pred > 0).long() == y_val).sum().item() / y_pred.shape[0] * 100
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.96, 0.96, f"Acc: {val_acc:.1f}%", horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    if not one_bar:
        cbar = plt.colorbar(mp, ax=ax)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
if one_bar:
#  add only one cbar for all subplots
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.005, 0.025, 0.015, .91])
    mp = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(vmin=-1, vmax=1))
    fig.colorbar(mp, cax=cbar_ax)
fig.tight_layout()
plt.savefig(f"LipLosses_onebar{one_bar}.pdf", bbox_inches="tight", pad_inches=0, transparent=True, dpi=200)
# %%
# y_pred = model_dhinge(x_train)
# loss = DynamicHingeLoss(margin=0.01, x=x_train, y_true=y_train)
# print(loss(y_pred, y_train))
# sign = 2 * y_train - 1
# print((loss.margin - sign * y_pred).relu().mean())
# print((loss.dynamic_margin + loss.margin - sign * y_pred).relu().mean())

# %%
