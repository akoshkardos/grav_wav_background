"""!@file GWB.py
@date 2024-07-26
@brief This program calculates the GWB based on the method described in my thesis, using uniform redshift bins.
@details The program calculates the GWB based on the method described in my thesis, using uniform redshift bins. It is divided into three main parts: the bulk part, the birth part, and the merger part. The bulk part calculates the majority of the GWB, what is referred to in my thesis as the 'generic case'. The birth part adds the contribution of the 'birth bins' to the bulk GWB. The merger part adds the contribution of the 'merger bins' due to Kepler max to the bulk GWB. The program saves a dataframe with all the essential information.
@author Seppe Staelens
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path
from Data_visualisation import calc_parabola_vertex, LISA_PI_curve, ET_PI_curve, Omega
import math

from astropy import units as u
from astropy import constants as cst
from astropy.cosmology import Planck18
from astropy.cosmology import z_at_value
from scipy.special import jv
from scipy.optimize import brentq
import seaborn as sns
import schnell as snl
import astropy.units as u
import matplotlib as mpl


plt.rc('font',   size=12)      # controls default text sizes
plt.rc('axes',   titlesize=14) # fontsize of the axes title
plt.rc('axes',   labelsize=14) # fontsize of the x and y labels
plt.rc('xtick',  labelsize=14) # fontsize of the tick labels
plt.rc('ytick',  labelsize=14) # fontsize of the tick labels
plt.rc('legend', fontsize=14) # legend fontsize
plt.rc('figure', titlesize=14) # fontsize of the figure title

# LVK O3, alpha = 2/3 at 25 Hz
Omega_upp = 2.9e-9
Omega_BBH = 0.8e-9
Omega_BNS = 3.6e-11

Omega_BBH_up  = 1.9e-9
Omega_BNS_up  = 1.0e-10
Omega_BBH_low = 0.3e-9
Omega_BNS_low = 0.8e-11

"""
Physics functions
"""

def Period(a: float, m1: float, m2: float) -> np.array:
    return (
        (a * cst.R_sun) ** 3 * 4 * np.pi**2 / (cst.G * cst.M_sun * (m1 + m2))
    ) ** (1 / 2)

def chirp_mass(m1, m2):
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

def F_e(e):
    return (1 + (73/24)*e**2 + (37/96)*e**4) / (1 - e**2)**(7/2)

def g_ne(n, e):

    Jn2 = jv(n-2, n*e)
    Jn1 = jv(n-1, n*e)
    Jn = jv(n, n*e)
    Jp1 = jv(n+1, n*e)
    Jp2 = jv(n+2, n*e)

    term1 = (Jn2 - 2*e*Jn1 + (2/n)*Jn + 2*e*Jp1 - Jp2)**2
    term2 = (1 - e**2)*(Jn2 - 2*Jn + Jp2)**2
    term3 = (4/(3*n**2))*Jn**2

    return (n**4 / 32) * (term1 + term2 + term3)

def Phi_n(n, e):
    return (2/n)**(2/3) * g_ne(n, e) / F_e(e)



def e_at_fp_ratio(fp_ratio, e_0):
    """!
    @brief At every fp/fp0 step give back the corresponding e.
    @param fp_ratio: fp/fp0 value for a given step starting at 1.
    @param e_0: starting eccentricity.
    @return e corresponding to fp/fp0.
    """
    if fp_ratio <= 1.0:
        return e_0  # hasn't evolved yet

    def residual(e):
        term1 = (1 - e_0**2) / (1 - e**2)
        term2 = (e / e_0)**(12/19)
        term3 = ((1 + 121/304 * e**2) /
                (1 + 121/304 * e_0**2))**(870/2299)
        return (term1 * term2 * term3)**(-3/2) - fp_ratio

    try:
        if residual(1e-7) * residual(e_0 * (1 - 1e-9)) > 0:
            return 1e-7   # already circularised
        return brentq(residual, 1e-7, e_0 * (1 - 1e-9))
    except Exception:
        return np.nan


def freq_e_interp(e_0: float):
    """!
    @brief Create e corresponding to fp/fp0 table that goes below fp/fp0 = 1 for making smooth curve in phi over fr/fp0 graph.
    @param e_0: starting eccentricity
    @return Table of e corresponding to fp/fp0
    """

    e_range = np.linspace(0.9999, 1e-6, 5000)
    freq_ratio = np.zeros_like(e_range)

    for i, e in enumerate(e_range):
        term1 = (1 - e_0**2) / (1 - e**2)
        term2 = (e / e_0)**(12/19)
        term3 = ((1 + 121/304 * e**2) / (1 + 121/304 * e_0**2))**(870/2299)

        freq_ratio[i] = (term1 * term2 * term3)**(-3/2)

    freqtable = pd.DataFrame({
        "eccentricity": e_range,
        "f_p/f_p0": freq_ratio
    })

    return freqtable

def compute_phi_gwb_harmonic(population: pd.DataFrame, n_pole: int = 5,
                               n_e_steps: int = 100, n_fr_bins: int = 50, SAVE_FIG: bool = False):
    """!
    @brief For each binary calculate fr = n * fp0 * (fp/fp0) at every e (so fp) step and store corresponding phi_n, create chirp mass weighted phi over fr graph.
    @param population: population file containing e0, f0, m1 and m2.
    @param n_pole: number of calculated harmonics.
    @param n_e_steps: number of calculated e steps.
    @param n_fr_bins: number of fr storage bins.
    @return Chirp mass weighted phi correction over fr.
    """

    ecc = population['e'].values
    m_c = chirp_mass(population['m1'].values, population['m2'].values)
    f_p0 = (1.0 / Period(population['a'].values, population['m1'].values, population['m2'].values)).to(u.Hz).value

    fp_ratio_steps = np.logspace(0, 3, n_e_steps)

    collected_fr      = {n: [] for n in range(1, n_pole + 1)}
    collected_phi     = {n: [] for n in range(1, n_pole + 1)}
    collected_m_c  = {n: [] for n in range(1, n_pole + 1)}

    for i, (e_0, fp0, mc) in enumerate(zip(ecc, f_p0, m_c)):

        mc_fac = mc**(5/3)

        for fp_ratio in fp_ratio_steps:
            e_step = e_at_fp_ratio(fp_ratio, e_0)
            if np.isnan(e_step):
                continue
            # To retrieve f_r multiply f_p0 with the ratio and currect n harmonic
            fp_current = fp0 * fp_ratio
            for n in range(1, n_pole + 1):
                fr = n * fp_current
                phi_n_val = Phi_n(n, e_step)
                collected_fr[n].append(fr)
                collected_phi[n].append(phi_n_val)
                collected_m_c[n].append(mc_fac)

    for n in range(1, n_pole + 1):
        collected_fr[n]     = np.array(collected_fr[n])
        collected_phi[n]    = np.array(collected_phi[n])
        collected_m_c[n] = np.array(collected_m_c[n])

    # Create logspace for all found f_r with corresponding phi and chirp mass values
    all_fr_flat = np.concatenate([collected_fr[n] for n in range(1, n_pole + 1)])
    fr_min = all_fr_flat[all_fr_flat > 0].min()
    fr_max = all_fr_flat.max()
    log_edges   = np.linspace(np.log10(fr_min), np.log10(fr_max), n_fr_bins + 1)
    bin_centers = 10**((log_edges[:-1] + log_edges[1:]) / 2)

    mean_phi_per_n = {}

    for n in range(1, n_pole + 1):
        log_fr = np.log10(collected_fr[n])
        bin_idx = np.digitize(log_fr, log_edges) - 1
        mean_phi_n = np.full(n_fr_bins, np.nan)

        for b in range(n_fr_bins):
            bin = (bin_idx == b)
            if np.any(bin):
                # Mass-weighted average of phi in this frequency bin
                mean_phi_n[b] = (np.sum(collected_phi[n][bin] * collected_m_c[n][bin])
                                 / np.sum(collected_m_c[n][bin]))

        mean_phi_per_n[n] = mean_phi_n

    # Collect data for the sum over all harmonics
    mean_phi_total = np.nansum(
        np.stack([mean_phi_per_n[n] for n in range(1, n_pole + 1)], axis=0),
        axis=0
    )
    # Mask bins where ALL harmonics were NaN (no data at all)
    all_nan_mask = np.all(
        np.stack([np.isnan(mean_phi_per_n[n]) for n in range(1, n_pole + 1)], axis=0),
        axis=0
    )
    mean_phi_total[all_nan_mask] = np.nan

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = plt.cm.tab10(np.linspace(0, 0.5, n_pole))

    for n in range(1, n_pole + 1):
        ax.plot(bin_centers, mean_phi_per_n[n],
                color=colors[n - 1], lw=1.5, linestyle='--',
                label=f'$\\Phi_{n}$ (n={n})')

    ax.plot(bin_centers, mean_phi_total,
            'k-', lw=2.5, label='$\\Phi_\\mathrm{total}$')

    ax.axhline(1.0, color='r', linestyle=':', lw=1.2,
               label='circular ($\\Phi=1$)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$f_r$')
    ax.set_ylabel('$\\langle\\Phi_n\\rangle$ (Chirp mass weighted)')
    ax.set_xlim(1e-7, 1e-4)
    ax.set_ylim(1e-3, 1.1)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    if SAVE_FIG:
        fig.savefig(f"../output/Figures/phi_GWB.png")

def phi_curve(e_0: float, n_pole: int, SAVE_FIG: bool = False):
    """!
    @brief Create phi over fr/fp0 graph for individual harmonics and the sum phi_n.
    @param e_0: starting eccentricity.
    @param n_pole: number of harmonics to show.
    @return Graph of phi over fr/fp0.
    """
    table = freq_e_interp(e_0)
    e_vals = table["eccentricity"].values
    phi_total = np.zeros_like(e_vals)

    for n in range(1, n_pole+1):

        phi_n_vals = Phi_n(n, e_vals)
        table[f"Phi_{n}"] = phi_n_vals
        table[f"fr/fp0 {n}"] = table["f_p/f_p0"] * n
        phi_total += phi_n_vals

    fr_fp0_grid = np.logspace(0, 3, 1000)

    Phi_interp_list = []
    for n in range(1, n_pole+1):
        f_n = table[f"fr/fp0 {n}"].values
        Phi_n_vals = table[f"Phi_{n}"].values
        interp_func = interp1d(np.log10(f_n), np.log10(Phi_n_vals), kind='linear', bounds_error=False, fill_value=-np.inf)
        Phi_interp_list.append(10**interp_func(np.log10(fr_fp0_grid)))

    Phi_total_interp = np.sum(Phi_interp_list, axis=0)

    Phi_total_df = pd.DataFrame({
        "fr/fp0": fr_fp0_grid,
        "Phi_total": Phi_total_interp
    })

    fig, ax = plt.subplots(figsize=(5, 5))

    for n in range(1, n_pole+1):
        ax.plot(table[f"fr/fp0 {n}"],
                table[f"Phi_{n}"], linestyle='--', label=f'Phi_{n}')

    ax.plot(Phi_total_df["fr/fp0"], Phi_total_df["Phi_total"], 'k-', label='Phi_total')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10**0, 10**3)
    ax.set_ylim(1e-5, 1.25)
    ax.set_xlabel("f_r / f_p0")
    ax.set_ylabel("Phi")
    ax.legend()
    plt.tight_layout()
    plt.show()
    if SAVE_FIG:
        fig.savefig(f"../output/Figures/phi_curve.png")

def eccentricity_cloud(population, n_pole=5, n_e_steps=50, SAVE_FIG: bool = False):
    """!
    @brief Plot the density of eccentricity vs radiative frequency for the population.
    """
    ecc_initial = population['e'].values
    f_p0 = (1.0 / Period(population['a'].values, population['m1'].values, population['m2'].values)).to(u.Hz).value
    fp_ratio_steps = np.logspace(0, 3, n_e_steps)

    all_fr = []
    all_fp = []
    all_e = []

    for e_0, fp0 in zip(ecc_initial, f_p0):
        for fp_ratio in fp_ratio_steps:
            e_step = e_at_fp_ratio(fp_ratio, e_0)
            if np.isnan(e_step):
                continue

            fp_current = fp0 * fp_ratio
            all_fp.append(fp_current)
            for n in range(1, n_pole + 1):
                fr = n * fp_current
                all_fr.append(fr)
                all_e.append(e_step)

    all_fr = np.array(all_fr)
    all_fp = np.array(all_fp)
    all_e = np.array(all_e)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Radiative Frequency (Spread out by harmonics)
    hb1 = ax1.hexbin(all_fr, all_e, gridsize=50, bins='log', cmap='inferno', xscale='log', mincnt=1)
    ax1.set_xscale('log')
    ax1.set_xlabel('$f_r$')
    ax1.set_ylabel('$e$')

    # Plot 2: Orbital Frequency (The physical track of the binary)
    # Using all_e[::n_pole] correctly matches the length of all_fp
    hb2 = ax2.hexbin(all_fp, all_e[::n_pole], gridsize=50, bins='log', cmap='inferno', xscale='log', mincnt=1)
    ax2.set_xscale('log')
    ax2.set_xlabel('$f_e$') # Corrected Label
    ax2.set_ylabel('$e$')

    plt.colorbar(hb1, ax=ax1, label='$\log_{10}(\mathrm{Count})$')
    plt.colorbar(hb2, ax=ax2, label='$\log_{10}(\mathrm{Count})$')

    plt.tight_layout()
    plt.show()
    if SAVE_FIG:
        fig.savefig(f"../output/Figures/eccentricity_cloud.png")


def compare_plot(compare, tag, SFRD, save: bool = False, add_WD: bool = False):

    fig, ax = plt.subplots(figsize=(8, 6))
    z_list = ["z0001", "z001", "z005", "z01", "z02", "z03"]
    SFRD_list = ["MZ19", "LZ19", "HZ19", "LZ21", "HZ21"]
    tag_list = ["GA1T", "GA4T", "AA1T", "AA4T"]
    #tag_list = ["GA4T", "GA4Z"]
    totals = {}
    for type in (SFRD_list if compare == "SFRD" else tag_list):
        type_data = None
        for z in z_list:
            if compare == "SFRD":
                path = f"../output/GWBs/SFH6{type}_100_20_{z}_wmerge_{tag}_BH.txt"
            else:
                path = f"../output/GWBs/SFH6{SFRD}_100_20_{z}_wmerge_{type}_BH.txt"
            try:
                data = pd.read_csv(path)
                if type_data is None:
                    type_data = data[["f", "Om"]].copy()
                else:
                    type_data["Om"] += data["Om"]
            except Exception as e:
                print(f"Skipped {type}, {z}: {e}")
        totals[type] = type_data
        omega_25 = np.interp(25, type_data["f"].values, type_data["Om"].values)
        print(f"Omega at 25 Hz ({type}): {omega_25:.3e}")

    WD_data_accum = None
    if add_WD:
        for z in z_list:
            path_WD = f"../output/GWBs/SFH6{SFRD}_50_20_{z}_wmerge_{tag}_WD.txt"
            try:
                data_WD = pd.read_csv(path_WD)
                if WD_data_accum is None:
                    WD_data_accum = data_WD[["f", "Om"]].copy()
                else:
                    WD_data_accum["Om"] += data_WD["Om"]
            except Exception as e:
                print(f"Skipped WD {z}: {e}")
    iter_list = SFRD_list if compare == "SFRD" else tag_list
    colors = plt.cm.brg(np.linspace(0.5, 0.9, len(iter_list)))
    for i, (key, data) in enumerate(totals.items()):
        if data is None:
            continue
        ax.plot(data["f"], data["Om"], linewidth=1.3, label=f"{key}", color=colors[i])

    if add_WD and WD_data_accum is not None:
        ax.plot(WD_data_accum["f"], WD_data_accum["Om"], linewidth=1.0, linestyle="--", color="black", label = f"WD {tag} {SFRD}")

    a, b, c = calc_parabola_vertex(-3, -12, -2.5, -12.5, -2, -12)
    get_Omega_LISA = LISA_PI_curve(t_obs=4)
    get_Omega_ET = ET_PI_curve(t_obs=4)
    logf1_DeciCon10, logpi_DeciCon10 = np.loadtxt(f"../data/DeciCon10.dat").transpose()

    log_f_high = 2 ; log_f_low = -5
    freq = np.logspace(log_f_low - 0.1, log_f_high + 0.1, 1000)

    ax.plot(freq, Omega(Omega_upp, 25, freq),
            linestyle=":", color="#00A6FB", linewidth=1.1,
            label="LVK Upper Limit", zorder=1)
    ax.plot(freq, Omega(Omega_BBH, 25, freq),
            color="#4A7FA8", linewidth=0.9, alpha=0.8,
            label="LVK BBH", zorder=1)
    ax.plot(freq, Omega(Omega_BNS, 25, freq),
            color="#1A4F78", linewidth=0.9, alpha=0.8,
            label="LVK BNS", zorder=1)

    ax.fill_between(freq, Omega(Omega_BBH_low, 25, freq), Omega(Omega_BBH_up, 25, freq),
                    color="#4A7FA8", alpha=0.16, zorder=0)
    ax.fill_between(freq, Omega(Omega_BNS_low, 25, freq), Omega(Omega_BNS_up, 25, freq),
                    color="#1A4F78", alpha=0.16, zorder=0)

    ax.plot(freq, get_Omega_LISA(freq),
            color="#AAAAAA", linewidth=1.0, linestyle=(0, (4, 2, 1, 2)),
            label="LISA", zorder=2)
    ax.plot(freq, get_Omega_ET(freq),
            color="#666666", linewidth=1.0, linestyle=(0, (4, 2, 1, 2)),
            label="ET", zorder=2)
    ax.loglog(10**logf1_DeciCon10, 10**logpi_DeciCon10,
            color="#333333", linewidth=1.0, linestyle=(0, (4, 2, 1, 2)),
          label="DECIGO", zorder=2)

    ax.set_xlim(10**(log_f_low), 10**(log_f_high))
    ax.set_ylim(1e-13, 1e-8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(f\,/\,\mathrm{Hz})$")
    ax.set_ylabel(r"$\Omega_\mathrm{GW}$")

    x_tick_vals = [10**i for i in range(int(np.floor(log_f_low)), int(log_f_high) + 1)]
    ax.set_xticks(x_tick_vals)
    ax.set_xticklabels([str(int(np.log10(v))) for v in x_tick_vals])

    ax.grid(which="major", linestyle="--", color="#DDDDDD", linewidth=0.6, alpha=0.8)
    ax.grid(which="minor", linestyle=":",  color="#EEEEEE", linewidth=0.4, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
          loc='upper left',
          fontsize=11, framealpha=0.6,
          edgecolor="#CCCCCC", borderpad=0.6)

    plt.tight_layout()
    if save:
        if compare == "SFRD":
            fig.savefig(f"../output/Figures/comparison_plot_{compare}_{tag}_BH.png", dpi=300)
        else:
            fig.savefig(f"../output/Figures/comparison_plot_{compare}_{SFRD}_BH.png", dpi=300)
    plt.show()


def GWB_uncertainty_plot(upper_SFRD, upper_tag, lower_SFRD, lower_tag, save: bool = True):

    fig, ax = plt.subplots(figsize=(8, 6))
    z_list = ["z0001", "z001", "z005", "z01", "z02", "z03"]
    data_upp_accum = None
    data_lower_accum = None
    for z in z_list:
        path_upp = f"../output/GWBs/SFH6{upper_SFRD}_100_20_{z}_wmerge_{upper_tag}_BH.txt"
        path_lower = f"../output/GWBs/SFH6{lower_SFRD}_100_20_{z}_wmerge_{lower_tag}_BH.txt"
        try:
            data_upp = pd.read_csv(path_upp)
            data_lower = pd.read_csv(path_lower)
            if data_upp_accum is None and data_lower_accum is None:
                data_upp_accum = data_upp[["f", "Om"]].copy()
                data_lower_accum = data_lower[["f", "Om"]].copy()
            else:
                data_upp_accum["Om"] += data_upp["Om"]
                data_lower_accum["Om"] += data_lower["Om"]
        except Exception as e:
            print(f"Skipped: {e}")


    f_arr = data_upp_accum["f"].values
    Om_arr_mid = 0.5 * (data_upp_accum["Om"].values + data_lower_accum["Om"].values)
    mask = (f_arr >= 1e-4) & (f_arr <= 1e-1)
    coeffs, cov = np.polyfit(np.log10(f_arr[mask]), np.log10(Om_arr_mid[mask]), 1, cov=True)
    slope_err = np.sqrt(cov[0, 0])
    alpha_err = np.sqrt(cov[1, 1])
    print(f"Slope: {coeffs[0]:.6f} ± {slope_err:.6f}")
    print(f"log10(alpha): {coeffs[1]:.3f} ± {alpha_err:.3f}")


    a, b, c = calc_parabola_vertex(-3, -12, -2.5, -12.5, -2, -12)
    get_Omega_LISA = LISA_PI_curve(t_obs=4)
    get_Omega_ET = ET_PI_curve(t_obs=4)
    logf1_DeciCon10, logpi_DeciCon10 = np.loadtxt(f"../data/DeciCon10.dat").transpose()

    log_f_high = 3.5 ; log_f_low = -6
    freq = np.logspace(log_f_low - 0.1, log_f_high + 0.1, 1000)

    ax.plot(freq, Omega(Omega_upp, 25, freq),
            linestyle="-", color="#B8D4E8", linewidth=0.9,
            label="LVK Upper Limit", zorder=1)
    ax.plot(freq, Omega(Omega_BBH, 25, freq),
            color="#4A7FA8", linewidth=0.9, alpha=0.8,
            label="LVK BBH", zorder=1)
    ax.plot(freq, Omega(Omega_BNS, 25, freq),
            color="#1A4F78", linewidth=0.9, alpha=0.8,
            label="LVK BNS", zorder=1)

    ax.plot(data_upp_accum["f"], data_upp_accum["Om"], linewidth=0.3, color="#C0392B", label="_nolegend_")
    ax.plot(data_lower_accum["f"], data_lower_accum["Om"], linewidth=0.3, color="#C0392B", label="_nolegend_")
    ax.fill_between(data_upp_accum["f"], data_lower_accum["Om"], data_upp_accum["Om"],
                    color="#C0392B", alpha=0.2, zorder=0, label="Uncertainty range")
    ax.fill_between(freq, Omega(Omega_BBH_low, 25, freq), Omega(Omega_BBH_up, 25, freq),
                    color="#4A7FA8", alpha=0.16, zorder=0)
    ax.fill_between(freq, Omega(Omega_BNS_low, 25, freq), Omega(Omega_BNS_up, 25, freq),
                    color="#1A4F78", alpha=0.16, zorder=0)

    ax.plot(freq, get_Omega_LISA(freq),
            color="#AAAAAA", linewidth=1.0, linestyle=(0, (4, 2, 1, 2)),
            label="LISA", zorder=2)
    ax.plot(freq, get_Omega_ET(freq),
            color="#666666", linewidth=1.0, linestyle=(0, (4, 2, 1, 2)),
            label="ET", zorder=2)
    ax.loglog(10**logf1_DeciCon10, 10**logpi_DeciCon10,
            color="#333333", linewidth=1.0, linestyle=(0, (4, 2, 1, 2)),
          label="DECIGO", zorder=2)

    ax.set_xlim(10**(log_f_low), 10**(log_f_high))
    ax.set_ylim(1e-15, 1e-8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(f\,/\,\mathrm{Hz})$")
    ax.set_ylabel(r"$\Omega_\mathrm{GW}$")

    x_tick_vals = [10**i for i in range(int(np.floor(log_f_low)), int(np.ceil(log_f_high)) + 1)]
    ax.set_xticks(x_tick_vals)
    ax.set_xticklabels([str(int(np.log10(v))) for v in x_tick_vals])

    ax.grid(which="major", linestyle="--", color="#DDDDDD", linewidth=0.6, alpha=0.8)
    ax.grid(which="minor", linestyle=":",  color="#EEEEEE", linewidth=0.4, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
          loc='upper left',
          fontsize=9, framealpha=0.85,
          edgecolor="#CCCCCC", borderpad=0.8)

    plt.tight_layout()
    if save:
        fig.savefig(f"../output/Figures/uncertainty_plot_BH.png", dpi=300)
    plt.show()

def density_initials(pop_synth, alpha, save: bool = False):
    z_dict = {"z0001": "Z = 0.0001", "z001": "Z = 0.001", "z005": "Z = 0.005",
              "z01": "Z = 0.01", "z02": "Z = 0.02", "z03": "Z = 0.03"}

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0)

    for ax, (z, z_label) in zip(axes.flat, z_dict.items()):
        initials = pd.read_csv(f"../data/BH/{pop_synth}/{alpha}/{z}/Initials_{z}.txt.gz", sep=",", header=0)
        sns.kdeplot(x=np.log10(2*initials.nu0), y=initials.M_ch,
                    fill=True, cmap="Blues", thresh=0.02, levels=15, ax=ax)
        ax.text(0.03, 0.97, z_label, transform=ax.transAxes, fontsize=20, va="top", ha="left")
        ax.set_xlim(-7, -3.5)
        ax.set_ylim(-3, 40)
        ax.grid(True, linestyle="--", color="#BBBBBB", linewidth=0.5, alpha=0.4)

    for ax in axes[1, :]:
        ax.set_xlabel(r"$\log_{10}(\nu_0/\mathrm{Hz})$", fontsize = 22)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\mathcal{M}_c\,/\,M_\odot$", fontsize = 22)

    plt.subplots_adjust(hspace=0, wspace=0)
    if save:
        fig.savefig(f"../output/Figures/initials_{pop_synth}_{alpha}.png", dpi=300)
    plt.show()


def density_t0_tau(pop_synth, alpha, save: bool = False):
    z_dict = {"z0001": "Z = 0.0001", "z001": "Z = 0.001", "z005": "Z = 0.005",
              "z01": "Z = 0.01", "z02": "Z = 0.02", "z03": "Z = 0.03"}

    t0_range = np.linspace(0, 20, 100)
    hubble_lines = {0: "-", 3: "--", 6: ":"}

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0)

    for ax, (z, z_label) in zip(axes.flat, z_dict.items()):
        initials = pd.read_csv(f"../data/BH/{pop_synth}/{alpha}/{z}/Initials_{z}.txt.gz", sep=",", header=0)
        sns.kdeplot(x=initials.t0, y=np.log10(initials.Dt_max),
                    fill=True, cmap="Reds", thresh=0.02, levels=20, ax=ax)
        for z_hub, ls in hubble_lines.items():
            t_hub = Planck18.age(z_hub).to(u.Myr).value
            valid = t0_range < t_hub
            ax.plot(t0_range[valid], np.log10(t_hub - t0_range[valid]), color="black", linewidth=0.8,
                    linestyle=ls, label=f"z = {z_hub}")
        ax.set_xlim(0, 20)
        ax.set_ylim(1, 9)
        ax.text(0.03, 0.97, z_label, transform=ax.transAxes, fontsize=20, va="top", ha="left")
        ax.grid(True, linestyle="--", color="#BBBBBB", linewidth=0.5, alpha=0.4)

    axes.flat[0].legend(fontsize=15, loc="upper right")

    for ax in axes[1, :]:
        ax.set_xlabel(r"$t_0$ [Myr]", fontsize=22)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}(\tau\,/\,\mathrm{Myr})$", fontsize=22)

    plt.subplots_adjust(hspace=0, wspace=0)
    if save:
        fig.savefig(f"../output/Figures/t0_tau_{pop_synth}_{alpha}.png", dpi=300)
    plt.show()

def density_nu0max(pop_synth, alpha, save: bool = False):
    z_dict = {"z0001": "Z = 0.0001", "z001": "Z = 0.001", "z005": "Z = 0.005",
              "z01": "Z = 0.01", "z02": "Z = 0.02", "z03": "Z = 0.03"}

    bins = np.linspace(1.5, 3.2, 50)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = np.diff(bins)

    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#9b2226"]
    bottoms = np.zeros(len(bin_centers))

    fig, ax = plt.subplots(figsize=(10, 6))
    for (z, z_label), color in zip(z_dict.items(), colors):
        initials = pd.read_csv(f"../data/BH/{pop_synth}/{alpha}/{z}/Initials_{z}.txt.gz", sep=",", header=0)
        counts, _ = np.histogram(np.log10(2 * initials.nu_max), bins=bins)
        ax.bar(bin_centers, counts, width=width, bottom=bottoms,
               color=color, label=z_label, alpha=0.9, linewidth=0)
        bottoms += counts

    ax.set_xlabel(r"$\log_{10}(\nu_{\rm max}\,/\,\mathrm{Hz})$", fontsize=22)
    ax.set_ylabel("Number of systems", fontsize=22)
    ax.grid(True, linestyle="--", color="#BBBBBB", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=15, framealpha=0.85, edgecolor="#CCCCCC")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(f"../output/Figures/nu_max_{pop_synth}_{alpha}.png", dpi=300)
    plt.show()

def main():
    data_file = Path(f"../data/BH/GammaAlpha/Alpha4/z02/t0aim1m1e.dat.gz")
    population = pd.read_csv(data_file, names=["t0", "a", "e", "m1", "m2"], sep="\s+")
    #mean_e_0 = np.mean(population["e"])
    #phi_curve(mean_e_0, 5, SAVE_FIG = True)
    #compute_phi_gwb_harmonic(population, n_pole=5, n_e_steps=100, n_fr_bins=50, SAVE_FIG = True)
    #eccentricity_cloud(population, SAVE_FIG = True)
    #compare_plot("tag", "AA4T","MZ19", True, True)
    #GWB_uncertainty_plot("LZ21", "GA1T", "HZ19", "GA4T", True)
    pop_synths = ["GammaAlpha", "AlphaAlpha"]
    alphas = ["Alpha4", "Alpha1"]
    for pop_synth in pop_synths:
        for alpha in alphas:
            #density_initials(pop_synth, alpha, True)
            #density_t0_tau(pop_synth, alpha, True)
            density_nu0max(pop_synth, alpha, True)
    #density_t0_tau("GammaAlpha", "Alpha4", True)
    #density_nu0max("GammaAlpha", "Alpha4", True)

main()