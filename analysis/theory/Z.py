import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import cmasher as cmr
from astropy.table import Table
import flare.plt as fplt

import sys
import os

from synthesizer.plt import single
# from synthesizer.filters import TopHatFilterCollection
from synthesizer.filters import FilterCollection
# from synthesizer.grid import SpectralGrid
from synthesizer.grid import Grid
# from synthesizer.binned.sfzh import SFH, ZH, generate_sfzh, generate_instant_sfzh
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh, generate_instant_sfzh
# from synthesizer.binned.galaxy import SEDGenerator
from synthesizer.galaxy.parametric import ParametricGalaxy as Galaxy
from unyt import yr, Myr


if __name__ == '__main__':

    # -------------------------------------------------
    # --- calculate the EW for a given line as a function of age

    grid_dir = '/cosma7/data/dp004/dc-seey1/modules/flares_xiion/analysis/theory'
    model = 'bpass-v2.2.1-bin_chab-300_cloudy-v17.03_log10Uref-2'
    grid = Grid(model, grid_dir=grid_dir)

    # --- define a filter collection including one TopHat filter mapped on to the same wavelength grid as the SPS model
    tophats = {'FUV': {'lam_min': 1400., 'lam_max': 1600.}}
    fc = FilterCollection(tophat_dict=tophats, new_lam=grid.lam)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 5))
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.95, wspace=0, hspace=0)

    handles = []

    for duration, ls in zip([10, 100, 1000], ['-', '--', '-.']):

        sfh = SFH.Constant({'duration': duration * Myr})

        norm = mpl.colors.Normalize(vmin=-5, vmax=-1.5)
        # cmap = cmr.get_sub_cmap('cmr.sunburst', 0.05, 0.85)
        # norm = mpl.colors.Normalize(vmin=-4, vmax=-2)
        cmap = cmr.get_sub_cmap('cmr.torch', 0.3, 0.9)

        xiion = []
        Q = []

        log10Zs = np.log10(grid.metallicities)

        for log10Z in log10Zs:

            color = cmap(norm(log10Z))

            Zh = ZH.deltaConstant({'log10Z': log10Z})  # constant metallicity

            # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
            sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh)

            # --- define galaxy object
            # galaxy = SEDGenerator(grid, sfzh)
            galaxy = Galaxy(sfzh)
            galaxy.get_stellar_spectra(grid)

            log10Q = grid.log10Q
            # print(log10Q)
            # print('keys:', log10Q.keys())
            Q_ = galaxy.get_Q(grid)  # get ionising photon number
            # Q_ = galaxy.get_Q()  # get ionising photon number

            sed = galaxy.spectra['stellar']

            luminosities = sed.get_broadband_luminosities(fc)

            LFUV = luminosities['FUV']

            xiion_ = Q_/LFUV
            Q.append(Q_)
            xiion.append(xiion_)
            ax1.scatter(log10Z, np.log10(Q_), color=color, zorder=2, s=10)
            ax2.scatter(log10Z, np.log10(xiion_), color=color, zorder=2, s=10)

        ax1.plot(log10Zs, np.log10(Q), lw=1, color='0.5', ls=ls,
                 zorder=1, label=rf'$\rm SF\ duration = {duration}\ Myr$')
        ax2.plot(log10Zs, np.log10(xiion), lw=1, color='0.5', ls=ls,
                 zorder=1)

    ax1.legend(fontsize=7, labelspacing=0.1)
    # ax1.set_ylabel(r'$\rm log_{10}(\dot{n}_{LyC}/s^{-1} M_{\odot}^{-1})$')
    ax1.set_ylabel(r'$\log_{10}[(\dot{N}_{\rm ion,intr}/M_\star)/\rm{s}^{-1}\rm{M}_{\odot}^{-1}]$')
    ax2.set_ylabel(r'$\log_{10}(\xi_{\rm ion}/\rm{erg^{-1}Hz})$')
    ax2.set_ylim([25., 26.])
    # ax2.set_ylabel(r'$\rm log_{10}(\xi_{ion}/erg^{-1}\ Hz)$')
    ax2.set_xlabel(r'$\rm log_{10}Z_{\star}$')
    ax1.grid(color='whitesmoke')
    ax1.set_axisbelow(True)
    ax2.grid(color='whitesmoke')
    ax2.set_axisbelow(True)

    fig.savefig('figs/theory_Z.pdf')
