

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
from synthesizer.grid import SpectralGrid, parse_grid_id
from synthesizer.binned.sfzh import SFH, ZH, generate_sfzh, generate_instant_sfzh
from synthesizer.binned.galaxy import SEDGenerator
from unyt import yr, Myr


if __name__ == '__main__':

    # -------------------------------------------------
    # --- calcualte the EW for a given line as a function of age

    norm = mpl.colors.Normalize(vmin=-5, vmax=-1.5)

    cmap = cmr.get_sub_cmap('cmr.sunburst', 0.05, 0.85)

    sfh = SFH.Constant({'duration': 10 * Myr})

    fig, ax = single()

    ax.axvline(2.35, c='k', alpha=0.1, lw=3)
    ax.axvline(2.3, c='k', alpha=0.1, lw=1)

    handles = []

    for sps_model in ['fsps', 'bpass-100', 'bpass-300']:

        if sps_model == 'bpass-100':
            suffix = r'\ m_{up}=100\ M_{\odot}'
            marker = 'o'
            ls = '-'
            a3s = np.array([2.0, 2.35, 2.7])
            models = [
                f'bpass-v2.2.1-bin_{slope}-100_cloudy-v17.03_log10Uref-2' for slope in ['100', '135', '170']]

        if sps_model == 'bpass-300':
            suffix = r'\ m_{up}=300\ M_{\odot}'
            marker = 'h'
            ls = '--'
            a3s = np.array([2.0, 2.35, 2.7])
            models = [
                f'bpass-v2.2.1-bin_{slope}-300_cloudy-v17.03_log10Uref-2' for slope in ['100', '135', '170']]

        if sps_model == 'fsps':
            suffix = ''
            marker = 'D'
            ls = ':'
            a3s = np.arange(1.5, 3.1, 0.1)
            # a3s = [1.5]
            models = [
                f'fsps-v3.2_imf3:{a3:.1f}_cloudy-v17.03_log10Uref-2' for a3 in a3s]

        model_info = parse_grid_id(models[0])

        for log10Z in [-4., -3., -2.]:  #

            color = cmap(norm(log10Z))

            if sps_model == 'fsps':
                handles.append(mlines.Line2D([], [], color=color, ls='-',
                                             lw=1, label=rf'$\rm Z=10^{{ {log10Z:.0f} }}$'))

            Zh = ZH.deltaConstant({'log10Z': log10Z})  # constant metallicity

            xiion = []

            for model in models:
                print(model)
                grid = SpectralGrid(model)

                # --- define a filter collection including one TopHat filter mapped on to the same wavelength grid as the SPS model
                fc = TopHatFilterCollection(
                    [('FUV', {'lam_min': 1400., 'lam_max': 1600})], new_lam=grid.lam)

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

            # ax1b.plot(a3s, np.log10(log10Q), lw=2, color=color, alpha=0.3,
            #           ls=ls, zorder=0, marker=marker, markersize=4)

            ax.plot(a3s, np.log10(xiion), lw=1, color=color,
                    ls=ls, zorder=1, marker=marker, markersize=3)

        handles.append(mlines.Line2D([], [], color='0.5', ls=ls, marker=marker, markersize=4,
                                     lw=1, label=rf'$\rm {model_info["sps_model"]}\ {model_info["sps_model_version"]} {suffix}$'))

    ax.set_ylim([25., 26.])
    ax.set_ylabel(r'$\rm log_{10}(\xi_{ion}/erg^{-1}\ Hz)$')
    ax.legend(handles=handles, fontsize=7, labelspacing=0.1)
    ax.set_xlabel(r'$\rm \alpha_{3}$')

    fig.savefig('figs/theory_imf.pdf')
