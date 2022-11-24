

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import cmasher as cmr
from astropy.table import Table

import sys
import os

from synthesizer.plt import single
from synthesizer.filters import TopHatFilterCollection
from synthesizer.grid import SpectralGrid
from synthesizer.binned.sfzh import SFH, ZH, generate_sfzh, generate_instant_sfzh
from synthesizer.binned.galaxy import SEDGenerator
from unyt import yr, Myr


if __name__ == '__main__':

    # -------------------------------------------------
    # --- calcualte the EW for a given line as a function of age

    model = 'bpass-v2.2.1-bin_chab-300_cloudy-v17.03_log10Uref-2'
    grid = SpectralGrid(model)

    # --- define a filter collection including one TopHat filter mapped on to the same wavelength grid as the SPS model
    fc = TopHatFilterCollection([('FUV', {'lam_min': 1400., 'lam_max': 1600})], new_lam=grid.lam)

    fig, ax = single()

    handles = []

    for duration, ls in zip([10, 100, 1000], ['-', '--', '-.']):

        sfh = SFH.Constant({'duration': duration * Myr})

        norm = mpl.colors.Normalize(vmin=-5, vmax=-1.5)

        cmap = cmr.get_sub_cmap('cmr.sunburst', 0.05, 0.85)

        xiion = []

        for log10Z in grid.log10Zs:  #

            color = cmap(norm(log10Z))

            Zh = ZH.deltaConstant({'log10Z': log10Z})  # constant metallicity

            # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
            sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh)

            # --- define galaxy object
            galaxy = SEDGenerator(grid, sfzh)

            Q = galaxy.get_Q()  # get ionising photon number

            sed = galaxy.spectra['stellar']

            luminosities = sed.get_broadband_luminosities(fc)

            LFUV = luminosities['FUV']

            xiion_ = Q/LFUV

            xiion.append(xiion_)
            ax.scatter(log10Z, np.log10(xiion_), color=color, zorder=2, s=10)

        ax.plot(grid.log10Zs, np.log10(xiion), lw=1, color='0.5', ls=ls,
                zorder=1, label=rf'$\rm SF\ duration = {duration}\ Myr$')

    ax.set_ylim([25., 26.])
    ax.set_ylabel(r'$\rm log_{10}(\xi_{ion}/erg^{-1}\ Hz)$')
    ax.legend(fontsize=7, labelspacing=0.1)
    ax.set_xlabel(r'$\rm log_{10}Z_{\star}$')

    fig.savefig('figs/theory_Z.pdf')
