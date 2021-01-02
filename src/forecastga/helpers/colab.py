#! /usr/bin/env python
# coding: utf-8
#
"""ForecastGA: Colab Utils"""

import matplotlib.pyplot as plt


def plot_colab(df, title=None, dark_mode=False):

    from IPython.display import Markdown as md

    plt.style.use("default")

    def show_md(txt):
        display(md(txt))

    if dark_mode:
        # Good all around color library
        plt.style.use("seaborn-colorblind")
        plt.rcParams.update(
            {
                "lines.color": "#565555",
                "legend.edgecolor": "#818080",
                "legend.borderpad": 0.6,
                "text.color": "white",
                "axes.facecolor": "#383838",
                "axes.edgecolor": "#565555",
                "axes.grid": True,
                "axes.labelcolor": "white",
                "grid.color": "#565555",
                "xtick.color": "white",
                "ytick.color": "white",
                "figure.facecolor": "#383838",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "white",
                "font.sans-serif": "Liberation Sans",
                "lines.linewidth": 2,
                "figure.figsize": [15, 10],
                "font.size": 16,
            }
        )

    else:
        plt.style.use("seaborn-colorblind")
        plt.rcParams.update(
            {
                "legend.borderpad": 0.6,
                "axes.grid": True,
                "font.sans-serif": "Liberation Sans",
                "lines.linewidth": 2,
                "figure.figsize": [15, 10],
                "font.size": 16,
            }
        )

    if title:
        show_md("## {}".format(title))

    df.plot()
    plt.show()
