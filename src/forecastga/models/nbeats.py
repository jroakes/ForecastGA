#! /usr/bin/env python
# coding: utf-8
#

"""NBEATS Model"""

import os
import numpy as np

import torch
from torch import optim
from torch.nn import functional as F
from nbeats_pytorch.model import (
    NBeatsNet,
)  # some import from the trainer script e.g. load/save functions.
import matplotlib.pyplot as plt

from base import BaseModel


CHECKPOINT_NAME = "nbeats-training-checkpoint.th"


## Forecast Model
class NBEATS_Model(BaseModel):
    """NBEATS Model Class"""

    def __init__(self):
        pass

    def dataframe(self, df, forecast_length, in_sample, device, train_portion=0.75):
        backcast_length = 1 * forecast_length

        df = df.values  # just keep np array here for simplicity.
        norm_constant = np.max(df)
        df = df / norm_constant  # small leak to the test set here.

        x_train_batch, y = [], []

        for i in range(
            backcast_length + 1, len(df) - forecast_length + 1
        ):  # 25% to 75% so 50% #### Watch out I had to plus one.
            x_train_batch.append(df[i - backcast_length : i])
            y.append(df[i : i + forecast_length])

        x_train_batch = np.array(x_train_batch)[..., 0]
        y = np.array(y)[..., 0]

        net = NBeatsNet(
            stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
            forecast_length=forecast_length,
            thetas_dims=[7, 8],
            nb_blocks_per_stack=3,
            backcast_length=backcast_length,
            hidden_layer_units=128,
            share_weights_in_stack=False,
            device=device,
        )

        if in_sample == True:
            c = int(len(x_train_batch) * train_portion)
            x_train, y_train = x_train_batch[:c], y[:c]
            x_test, y_test = x_train_batch[c:], y[c:]

            return x_train, y_train, x_test, y_test, net, norm_constant

        else:
            c = int(len(x_train_batch) * 1)
            x_train, y_train = x_train_batch[:c], y[:c]

            return x_train, y_train, net, norm_constant

    def forecast(self):
        """Forecast NEATS Model"""

        if in_sample:
            net = model["model"]
            x_test = model["x_test"]
            y_test = model["y_test"]
            norm_constant = model["constant"]
            net.eval()
            _, forecast = net(torch.tensor(x_test, dtype=torch.float))
            if GPU:
                p = forecast.cpu().detach().numpy()
            else:
                p = forecast.detach().numpy()
            self.forecast = p[-1] * norm_constant
        else:
            net = model["model"]
            net.eval()
            x_train, y_train, net, norm_constant = model["tuple"]
            _, forecast = net(torch.tensor(x_train, dtype=torch.float))
            if GPU:
                p = forecast.cpu().detach().numpy()
            else:
                p = forecast.detach().numpy()
            self.forecast = p[-1] * norm_constant


    def train(self):
        """Train NBEATS Model"""

        if GPU:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if os.path.isfile(CHECKPOINT_NAME):
            os.remove(CHECKPOINT_NAME)
        stepped = 35
        batch_size = 10
        if in_sample:
            x_train, y_train, x_test, y_test, net, norm_constant = nbeats_dataframe(
                full_df, forecast_len, in_sample=True, device=device
            )
            optimiser = optim.Adam(net.parameters())
            data = data_generator(x_train, y_train, batch_size)
            # test_losses = []
            for r in range(stepped):
                train_100_grad_steps(data, device, net, optimiser)  # test_losses

            self.model = {}
            self.model["model"] = net
            self.model["x_test"] = x_test
            self.model["y_test"] = y_test
            self.model["constant"] = norm_constant

        else:  # if out_sample train is df

            x_train, y_train, net, norm_constant = nbeats_dataframe(
                full_df, forecast_len, in_sample=False, device=device
            )

            batch_size = 10  # greater than 4 for viz
            optimiser = optim.Adam(net.parameters())
            data = data_generator(x_train, y_train, batch_size)
            stepped = 5
            # test_losses = []
            for r in range(stepped):
                train_100_grad_steps(data, device, net, optimiser)  # test_losses
            self.model = {}
            self.model["model"] = net
            self.model["tuple"] = (x_train, y_train, net, norm_constant)



## NBEATS UTILS

def plot_scatter(*args, **kwargs):
    """plot utils"""

    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


def data_generator(x_full, y_full, bs):
    """simple batcher"""

    def split(arr, size):
        arrays = []
        while len(arr) > size:
            slice_ = arr[:size]
            arrays.append(slice_)
            arr = arr[size:]
        arrays.append(arr)
        return arrays

    while True:
        for rr in split((x_full, y_full), bs):
            yield rr


def train_100_grad_steps(data, device, net, optimiser):
    """Trainer"""
    global_step = load(net, optimiser)
    for x_train_batch, y_train_batch in data:
        global_step += 1
        optimiser.zero_grad()
        net.train()
        _, forecast = net(torch.tensor(x_train_batch, dtype=torch.float).to(device))
        loss = F.mse_loss(
            forecast, torch.tensor(y_train_batch, dtype=torch.float).to(device)
        )
        loss.backward()
        optimiser.step()
        if global_step > 0 and global_step % 100 == 0:
            with torch.no_grad():
                save(net, optimiser, global_step)
            break


def load(model, optimiser):
    """loader/saver for checkpoints"""

    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        grad_step = checkpoint["grad_step"]
        # print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return grad_step
    return 0


def save(model, optimiser, grad_step):
    torch.save(
        {
            "grad_step": grad_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
        },
        CHECKPOINT_NAME,
    )


def eval_test(
    backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test
):
    """evaluate model on test data and produce some plots."""

    net.eval()
    _, forecast = net(torch.tensor(x_test, dtype=torch.float))
    test_losses.append(
        F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item()
    )
    p = forecast.detach().numpy()
    subplots = [221, 222, 223, 224]
    plt.figure(1)
    for plot_id, i in enumerate(
        np.random.choice(range(len(x_test)), size=4, replace=False)
    ):
        ff, xx, yy = (
            p[i] * norm_constant,
            x_test[i] * norm_constant,
            y_test[i] * norm_constant,
        )
        plt.subplot(subplots[plot_id])
        plt.grid()
        plot_scatter(range(0, backcast_length), xx, color="b")
        plot_scatter(
            range(backcast_length, backcast_length + forecast_length), yy, color="g"
        )
        plot_scatter(
            range(backcast_length, backcast_length + forecast_length), ff, color="r"
        )
    plt.show()
