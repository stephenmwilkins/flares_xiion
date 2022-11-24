

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr

import sys
import os

from synthesizer.grid import SpectralGrid


if __name__ == '__main__':

    model = 'bpass-v2.2.1-bin_chab-300_cloudy-v17.03_log10Uref-2'

    grid = SpectralGrid(model, verbose=True)

    fig, ax = grid.plot_log10Q()

    fig.savefig(f'figs/theory_log10Q.pdf')
