#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: NBEATS Model"""

import os
import numpy as np

import torch
from torch import optim
from torch.nn import functional as F
from nbeats_pytorch.model import (
    NBeatsNet,
)


from base import BaseModel

from forecastga.helpers.data import data_generator


CHECKPOINT_NAME = "nbeats-training-checkpoint.th"


## Forecast Model
class NBEATS_Model(BaseModel):
    """NBEATS Model Class"""

    def train(self, **kwargs):
        """Train NBEATS Model"""

        if os.path.isfile(CHECKPOINT_NAME):
            os.remove(CHECKPOINT_NAME)

        steps = kwargs.get("steps", 50)
        batch_size = kwargs.get("batch_size", 10)
        patience = kwargs.get("patience", 5)
        device = self.get_device()

        net = NBeatsNet(
            stack_types=[
                NBeatsNet.TREND_BLOCK,
                NBeatsNet.SEASONALITY_BLOCK,
                NBeatsNet.GENERIC_BLOCK,
            ],
            forecast_length=self.forecast_len,
            thetas_dims=kwargs.get("thetas_dims", [2, 8, 3]),
            nb_blocks_per_stack=kwargs.get("nb_blocks_per_stack", 3),
            backcast_length=self.forecast_len,
            hidden_layer_units=kwargs.get("hidden_layer_units", 128),
            share_weights_in_stack=False,
            device=device,
        )

        x_batch, y_batch, norm_constant = self.format_input(
            self.dataframe, self.forecast_len
        )

        tp = self.train_proportion if self.in_sample else 1
        c = int(len(x_batch) * tp)

        optimiser = optim.Adam(net.parameters())

        data = data_generator(x_batch[:c], y_batch[:c], batch_size)

        best_loss = float("inf")
        counter = 0

        for _ in range(steps):
            loss = self.train_100_grad_steps(data, device, net, optimiser)
            if loss < best_loss:
                best_loss = loss
            else:
                counter += 1
                if counter >= patience:
                    break

        self.model = net
        self.constant = norm_constant

    def forecast(self, **kwargs):
        """Forecast NEATS Model"""

        x_batch, _, _ = self.format_input(
            self.dataframe, self.forecast_len, constant=self.constant
        )

        tp = self.train_proportion if self.in_sample else 1
        c = int(len(x_batch) * tp)

        self.model.eval()

        if self.in_sample:
            _, forecast = self.model(torch.tensor(x_batch[c:], dtype=torch.float))
        else:
            _, forecast = self.model(torch.tensor(x_batch, dtype=torch.float))

        p = forecast.cpu().detach().numpy() if self.GPU else forecast.detach().numpy()
        self.prediction = p[-1] * self.constant

    def get_device(self):
        return torch.device("cuda") if self.GPU else torch.device("cpu")

    @staticmethod
    def format_input(df, forecast_length, constant=None):

        backcast_length = 1 * forecast_length

        x = df.values
        norm_constant = constant if constant else np.max(x)
        x = x / norm_constant

        x_batch, y_batch = [], []

        # Batches the results into x_train_batch: x and y_train_batch: x + forecast_length
        for i in range(backcast_length + 1, len(df) - forecast_length + 1):
            x_batch.append(df[i - backcast_length : i])
            y_batch.append(df[i : i + forecast_length])

        x_batch = np.array(x_batch)[..., 0]
        y_batch = np.array(y_batch)[..., 0]

        return x_batch, y_batch, norm_constant

    def train_100_grad_steps(self, data, device, net, optimiser):
        """Trainer"""
        global_step = self.load(net, optimiser)
        global_step_init = global_step
        step_loss = 0
        for x_train_batch, y_train_batch in data:
            global_step += 1
            optimiser.zero_grad()
            net.train()
            _, forecast = net(torch.tensor(x_train_batch, dtype=torch.float).to(device))
            loss = F.mse_loss(
                forecast, torch.tensor(y_train_batch, dtype=torch.float).to(device)
            )
            step_loss += loss.item()
            loss.backward()
            optimiser.step()
            if global_step > 0 and global_step % 100 == 0:
                with torch.no_grad():
                    self.save(net, optimiser, global_step)
                break

        return step_loss / (global_step - global_step_init)

    def load(self, model, optimiser):
        """loader/saver for checkpoints"""

        if os.path.exists(CHECKPOINT_NAME):
            checkpoint = torch.load(CHECKPOINT_NAME)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
            grad_step = checkpoint["grad_step"]
            # print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
            return grad_step
        return 0

    def save(self, model, optimiser, grad_step):
        torch.save(
            {
                "grad_step": grad_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
            },
            CHECKPOINT_NAME,
        )
