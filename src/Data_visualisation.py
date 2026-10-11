"""!
@file Data_visualisation.py
@date 2026-03-18
@brief Plot and summarize calculated gravitational-wave background results.
@details Produces spectrum plots, redshift-contribution charts, frequency and
redshift system-count distributions, and system-count summaries. Supports the
SN24 and HN24 example datasets.
@author Ákosh Kardos
"""

import configparser as cfg
import os
import sys
from pathlib import Path
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import schnell as snl
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value

# matplotlib globals
plt.rc('font',   size=12)      # controls default text sizes
plt.rc('axes',   titlesize=14) # fontsize of the axes title
plt.rc('axes',   labelsize=14) # fontsize of the x and y labels
plt.rc('xtick',  labelsize=14) # fontsize of the tick labels
plt.rc('ytick',  labelsize=14) # fontsize of the tick labels
plt.rc('legend', fontsize=14) # legend fontsize
plt.rc('figure', titlesize=14) # fontsize of the figure title

# LVK O3, alpha = 2/3 at 25 Hz
OMEGA_UPPER = 2.9e-9
OMEGA_BBH = 0.8e-9
OMEGA_BNS = 3.6e-11

OMEGA_BBH_UP  = 1.9e-9
OMEGA_BNS_UP  = 1.0e-10
OMEGA_BBH_LOW = 0.3e-9
OMEGA_BNS_LOW = 0.8e-11

# --- Plotting and spectrum helpers ---

def lisa_pi_curve(t_obs: float = 4):
    """!
    @brief Build a callable for the LISA power-law-integrated sensitivity curve.
    @param t_obs: Observation time in years; defaults to 4.
    @return Function that accepts frequencies in Hz and returns the PI curve.
    @details Based on a script by Gijs Nelemans.
    """

    f_ref = 1e-2
    nside = 1

    dets = [snl.LISADetector(i) for i in range(3)]
    rho = snl.NoiseCorrelationLISA(dets[0])
    mc = snl.MapCalculator(dets, f_pivot=f_ref, corr_matrix=rho)

    obs_time = t_obs * 365 * 24 * 3600

    def get_omega_lisa(freqs):
        pi_LISA = mc.get_pi_curve(
            obs_time,
            freqs,
            nside,
            is_fspacing_log=True,
            nsigma=2,
            proj={'vectors': np.ones(3), 'deproject': True}
        )
        return pi_LISA

    return get_omega_lisa


def et_pi_curve(t_obs: float = 4):
    """!
    @brief Build a callable for the Einstein Telescope PI sensitivity curve.
    @param t_obs: Observation time in years; defaults to 4.
    @return Function that accepts frequencies in Hz and returns the PI curve.
    @details Uses the ET-D sum sensitivity and ../data/ET.txt.
    """

    path = f"../data/ET.txt"

    f_ref = 63.
    nside = 1

    et_dets = [
        snl.GroundDetectorTriangle(
            name=f"ET{i}", lat=40.1, lon=9.0, fname_psd=path, detector_id=i
        )
        for i in range(3)
    ]

    mc_ET = snl.MapCalculator(et_dets, f_pivot=f_ref, spectral_index=2./3.)
    obs_time = t_obs * 365 * 24 * 3600

    def get_omega_et(freqs):
        pi_ET = mc_ET.get_pi_curve(
            obs_time,
            freqs,
            nside,
            no_autos=True,
            nsigma=4
        )
        return pi_ET

    return get_omega_et


def get_z_list_bins_from_time(max_z: float, n_int: int) -> tuple:
    """!
    @brief Convert uniform lookback-time samples and edges to redshift.
    @param max_z: Maximum redshift defining the lookback-time range.
    @param n_int: Number of integration intervals.
    @return Pair of redshift sample points and bin edges.
    """

    max_lookback_time = cosmo.lookback_time(max_z)
    t_range = np.linspace(0, max_lookback_time.value, 2 * n_int + 1)

    t_list = np.array([t_range[2 * i + 1] for i in range(n_int)])
    t_bins = np.array([t_range[2 * i] for i in range(n_int + 1)])

    z_list = []
    z_bins = []

    for time in t_list:
        z_list.append(z_at_value(cosmo.lookback_time, time * u.Gyr).value)

    for time in t_bins[1:]:
        z_bins.append(z_at_value(cosmo.lookback_time, time * u.Gyr).value)

    return z_list, z_bins


def omega(omega_ref: float, f_ref: float, freq: np.ndarray) -> np.ndarray:
    """!
    @brief Evaluate a spectrum normalized to omega_ref at f_ref with slope 2/3.
    @param omega_ref: Spectrum value at the reference frequency.
    @param f_ref: Reference frequency in Hz.
    @param freq: Frequencies in Hz.
    @return Spectrum values at freq.
    """
    return omega_ref * (freq / f_ref)**(2/3)


def parabola(freq: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """!
    @brief Evaluate a quadratic polynomial.
    @param freq: x values at which to evaluate the polynomial.
    @param a: Quadratic coefficient.
    @param b: Linear coefficient.
    @param c: Constant coefficient.
    @return Polynomial values at freq.
    """
    return a * freq**2 + b * freq + c


def load_data(
    sfh: int,
    sfh_type: str,
    n_freq: int,
    n_int: int,
    tag: str,
    metallicity: Union[str, list[str]],
    compact_object: str,
    subfolder: str = "",
    include_num: bool = True,
):
    """!
    @brief Load contribution tables and optionally combine metallicities.
    @param sfh: Numeric star-formation-history selector.
    @param sfh_type: Tabulated history label, if used.
    @param n_freq: Number of frequency bins in the output files.
    @param n_int: Number of integration bins in the output files.
    @param tag: Output run tag.
    @param metallicity: One metallicity key or a sequence of keys.
    @param compact_object: Population type label, such as WD or BH.
    @param subfolder: Output subfolder or legacy dataset selector (SN24/HN24).
    @param include_num: Sum number-count columns when combining datasets;
    otherwise sum frequency-contribution columns.
    @return Three dataframes for bulk, birth, and merger contributions, or
    (None, None, None) if no matching files were found.
    """
    z_list = [metallicity] if isinstance(metallicity, str) else metallicity
    d1_total, d2_total, d3_total = None, None, None

    for z in z_list:
        try:
            if subfolder == "SN24":
                path_base = f"../output/GWBs/SN24/SFH1_50_20_z_contr"
                d1 = pd.read_csv(f"{path_base}_{tag}.txt.gz")
                d2 = pd.read_csv(f"{path_base}_birth_{tag}.txt.gz")
                d3 = pd.read_csv(f"{path_base}_merge_{tag}.txt.gz")
            elif subfolder == "HN24":
                path_base = f"../output/GWBs/HN24/SFH6_50_20_{z}_z_contr"
                d1 = pd.read_csv(f"{path_base}_example_HN24.txt.gz")
                d2 = pd.read_csv(f"{path_base}_birth_example_HN24.txt.gz")
                d3 = pd.read_csv(f"{path_base}_merge_example_HN24.txt.gz")
            else:
                path_base = (
                    f"../output/GWBs{subfolder}/SFH{sfh}{sfh_type}_"
                    f"{n_freq}_{n_int}_{z}_z_contr"
                )
                d1 = pd.read_csv(f"{path_base}_{tag}_{compact_object}.txt")
                d2 = pd.read_csv(f"{path_base}_birth_{tag}_{compact_object}.txt")
                d3 = pd.read_csv(f"{path_base}_merge_{tag}_{compact_object}.txt")

            if d1_total is None:
                d1_total, d2_total, d3_total = d1.copy(), d2.copy(), d3.copy()
            else:
                # Determine which columns to sum
                if include_num:
                    # For summary and distr: needs num in the column name
                    cols = [c for c in d1.columns if "freq" in c and "num" in c]
                else:
                    # For stacked bar: only needs freq (the Om values)
                    cols = [c for c in d1.columns if "freq" in c and "num" not in c]

                d1_total[cols] += d1[cols]
                d2_total[cols] += d2[cols]
                d3_total[cols] += d3[cols]
            if subfolder == "SN24":
                break
        except FileNotFoundError:
            print(f"Data Loader skipping {z}: Files not found at {path_base}")

    return d1_total, d2_total, d3_total


# --- Spectrum and contribution visualizations ---

def multiplot(
    sfh: int,
    sfh_type: str,
    n_freq: int,
    n_int: int,
    tag: str,
    compact_object: str,
    save: bool = False,
    save_name = "",
    compare: bool = False,
    z_list: list[str] = ["z0001", "z0005", "z001", "z005", "z01", "z02", "z03"],
    subfolder: str = ""
) -> None:
    """!
    @brief Plot GWB spectra for selected metallicities with detector curves.
    @param sfh: Numeric star-formation-history selector.
    @param sfh_type: Tabulated history label, if used.
    @param n_freq: Number of frequency bins represented by the spectra.
    @param n_int: Number of integration bins represented by the spectra.
    @param tag: Output run tag.
    @param compact_object: Population type label, such as WD or BH.
    @param save: Save the figure when true.
    @param save_name: Text included in the saved figure filename.
    @param compare: Include a comparison spectrum for the other population type.
    @param z_list: Metallicities whose spectra are plotted.
    @param subfolder: Output subfolder or legacy dataset selector (SN24/HN24).
    @details Displays the figure and saves it under ../output/Figures when save is true.
    """

    get_omega_lisa = lisa_pi_curve(t_obs=4)

    get_omega_et = et_pi_curve(t_obs=4)

    # DECIGO Data
    logf1_DeciCon10, logpi_DeciCon10 = np.loadtxt(f"../data/DeciCon10.dat").transpose()

    fig, ax = plt.subplots(figsize=(8, 6))

    metal_colors = ["#f0b990", "#eda070", "#ea8a55", "#e67840", "#e36830", "#df5820", "#db4810"]
    total_data = None

    for i, z in enumerate(z_list):
        ls, alpha = "-", 1.0
        color = metal_colors[i % len(metal_colors)]
        if subfolder == "SN24":
            path = f"../output/GWBs/{subfolder}/SFH1_50_20_wmerge_{tag}.txt.gz"
            data = pd.read_csv(path)
            compare = False
            ax.plot(data.f, data.Om, color=color, linestyle=ls, alpha=alpha, label="SN24")
            break
        elif subfolder == "HN24":
            path = f"../output/GWBs/{subfolder}/SFH6_50_20_{z}_wmerge_example_HN24.txt.gz"
            sfh = 6
            compare = False
        else:
            path = (
                f"../output/GWBs/SFH{sfh}{sfh_type}_{n_freq}_{n_int}_{z}_"
                f"wmerge_{tag}_{compact_object}.txt"
            )
        try:
            data = pd.read_csv(path)
            if sfh == 6:
                if total_data is None:
                    total_data = data.copy()
                else:
                    total_data["Om"] += data["Om"]
                ax.plot(data.f, data.Om,
                        color=color, linestyle="--",
                        linewidth=0.9, alpha=0.35, label=z)
            else:
                ax.plot(data.f, data.Om,
                        color=color, linestyle=ls,
                        linewidth=2.0, alpha=0.4, label=z)
        except Exception as e:
            print(f"Failed to load z={z}. Error was: {type(e).__name__}: {e}")

    if sfh == 6 and total_data is not None:
        ax.plot(total_data.f, total_data.Om,
                color="#E8622A",
                linewidth=3.0,
                zorder=10,
                label="Total GWB")

    compare_data = None
    if compact_object == "BH":
        log_f_high = 3.5 ; log_f_low = -6 ; comp_obj = "WD"
        if compare:
            path = f"../output/GWBs/SFH{sfh}{sfh_type}_{n_freq}_{n_int}_z01_wmerge_{tag}_WD.txt"
            if not os.path.exists(path):
                path = (
                    f"../output/GWBs/SFH{sfh}{sfh_type}_50_{n_int}_"
                    f"z01_wmerge_{tag}_WD.txt"
                )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"WD run with configs {tag}, SFH{sfh}, and n_freq=50 "
                    "hasn't been done for middle z(01)."
                )
            compare_data = pd.read_csv(path)
    else:
        log_f_high = 0 ; log_f_low = -5 ; comp_obj = "BH"
        if compare:
            path = f"../output/GWBs/SFH{sfh}{sfh_type}_100_{n_int}_z001_wmerge_{tag}_BH.txt"
            if not os.path.exists(path):
                path = (
                    f"../output/GWBs/SFH{sfh}{sfh_type}_{n_freq}_{n_int}_"
                    f"z001_wmerge_{tag}_BH.txt"
                )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"BH run with configs {tag}, SFH{sfh}, and n_freq=100 "
                    "hasn't been done for middle z(001)."
                )
            compare_data = pd.read_csv(path)

    if compare_data is not None:
        ax.plot(compare_data.f, compare_data.Om,
                color="#C0392B", linestyle="--", linewidth=1.8,
                label=f"Comp: {comp_obj} (z01)")

    freq = np.logspace(log_f_low - 0.1, log_f_high + 0.1, 1000)


    ax.plot(freq, omega(OMEGA_UPPER, 25, freq),
            linestyle="-", color="#B8D4E8", linewidth=0.9,
            label="LVK Upper Limit", zorder=1)
    ax.plot(freq, omega(OMEGA_BBH, 25, freq),
            color="#4A7FA8", linewidth=0.9, alpha=0.8,
            label="LVK BBH", zorder=1)
    ax.plot(freq, omega(OMEGA_BNS, 25, freq),
            color="#1A4F78", linewidth=0.9, alpha=0.8,
            label="LVK BNS", zorder=1)

    ax.fill_between(freq, omega(OMEGA_BBH_LOW, 25, freq), omega(OMEGA_BBH_UP, 25, freq),
                    color="#4A7FA8", alpha=0.16, zorder=0)
    ax.fill_between(freq, omega(OMEGA_BNS_LOW, 25, freq), omega(OMEGA_BNS_UP, 25, freq),
                    color="#1A4F78", alpha=0.16, zorder=0)


    ax.plot(freq, get_omega_lisa(freq),
            color="#AAAAAA", linewidth=1.0, linestyle=(0, (4, 2, 1, 2)),
            label="LISA", zorder=2)
    ax.plot(freq, get_omega_et(freq),
            color="#666666", linewidth=1.0, linestyle=(0, (4, 2, 1, 2)),
            label="ET", zorder=2)
    ax.loglog(10**logf1_DeciCon10, 10**logpi_DeciCon10,
            color="#333333", linewidth=1.0, linestyle=(0, (4, 2, 1, 2)),
            label="DECIGO", zorder=2)


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10**(log_f_low), 10**(log_f_high))
    if compare or compact_object == "BH":
        ax.set_ylim(1e-15, 1e-8)
    else:
        ax.set_ylim(1e-14, 1e-10)

    ax.set_xlabel(r"$\log_{10}(f\,/\,\mathrm{Hz})$")
    ax.set_ylabel(r"$\Omega_{\mathrm{GW}}$")


    x_tick_vals = [10**i for i in range(int(np.floor(log_f_low)), int(np.ceil(log_f_high)) + 1)]
    ax.set_xticks(x_tick_vals)
    ax.set_xticklabels([str(int(np.log10(v))) for v in x_tick_vals])

    ax.grid(which="major", linestyle="--", color="#DDDDDD", linewidth=0.6, alpha=0.8)
    ax.grid(which="minor", linestyle=":",  color="#EEEEEE", linewidth=0.4, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              loc='upper left',
              fontsize=11, framealpha=0.6,
              edgecolor="#CCCCCC", borderpad=0.6)

    if save:
        plt.tight_layout()
        fig.savefig(
            f"../output/Figures/multi_{n_freq}_{n_int}_SFH{sfh}{sfh_type}_"
            f"{tag}_{compact_object}_{save_name}{subfolder}.png",
            dpi=300,
        )

    plt.show()


def stacked_bar_chart(log_f_low, log_f_high, sfh, sfh_type, n_freq,
                      n_int, tag, metallicity, compact_object, integration_mode,
                      max_z=8, save=False, z_threshold=0.46, subfolder=""):
    """!
    @brief Plot the fractional contribution by integration/redshift bin per frequency.
    @param log_f_low: Base-10 logarithm of the lower frequency bound in Hz.
    @param log_f_high: Base-10 logarithm of the upper frequency bound in Hz.
    @param sfh: Numeric star-formation-history selector.
    @param sfh_type: Tabulated history label, if used.
    @param n_freq: Number of frequency bins.
    @param n_int: Number of integration bins.
    @param tag: Output run tag.
    @param metallicity: One metallicity key or sequence of keys to combine.
    @param compact_object: Population type label, such as WD or BH.
    @param integration_mode: Either redshift or time, determining bin construction.
    @param max_z: Maximum redshift defining the integration range.
    @param save: Save the figure when true.
    @param z_threshold: Redshift threshold marked on the chart.
    @param subfolder: Output subfolder or legacy dataset selector.
    @details Displays the stacked chart and optionally saves it under output/Figures.
    """
    d1, d2, d3 = load_data(
        sfh, sfh_type, n_freq, n_int, tag, metallicity, compact_object,
        subfolder, include_num=False,
    )
    if d1 is None:
        print("Stacked Bar Chart: No data found.")
        return

    f_range = np.logspace(log_f_low, log_f_high, 2*n_freq+1, base=10)
    f_plot = np.log10(np.array([f_range[2*i+1] for i in range(n_freq)]))

    if integration_mode == "redshift":
        z_bins = np.linspace(0, max_z, n_int + 1)
    if integration_mode == "time":
        _, z_bins = get_z_list_bins_from_time(max_z, n_int)

    z_bins = np.array(z_bins)


    prop_cols = {}
    for i in range(n_freq):
        total_f = d1[f"freq_{i}"] + d2[f"freq_{i}"] + d3[f"freq_{i}"]
        t_sum = total_f.sum()
        prop_cols[f"prop_f_{i}"] = (99 * total_f / t_sum) if t_sum != 0 else np.zeros(len(d1))

    new_data = pd.DataFrame(prop_cols)
    mask = new_data.sum(axis=0).values > 0
    f_filtered = f_plot[mask]
    data_filtered = new_data.loc[:, mask]
    rows_per_bin = len(data_filtered) // n_int


    ccmap = plt.get_cmap('viridis')
    if integration_mode == "time":
        # Ensure no zeros for LogNorm
        z_safe = np.where(z_bins <= 0, 1e-4, z_bins)
        norm = mpl.colors.LogNorm(vmin=z_safe.min(), vmax=z_safe.max())
    if integration_mode == "redshift":
        norm = mpl.colors.Normalize(vmin=z_bins.min(), vmax=z_bins.max())


    fig, ax = plt.subplots(figsize=(9, 6))
    width = (f_plot[1] - f_plot[0]) * 0.8 if len(f_plot) > 1 else 0.1

    current_sum = np.zeros(len(f_filtered))
    draw_line = False


    for i in range(n_int):
        start = i * rows_per_bin
        end = start + rows_per_bin
        props = data_filtered.iloc[start:end].sum().values

        ax.bar(f_filtered, props, bottom=current_sum,
               color=ccmap(norm(z_bins[i])), width=width)

        # Threshold check (ensure i+1 is within bounds)
        if not draw_line and (i + 1 < len(z_bins)) and (z_bins[i+1] >= z_threshold):
            draw_line = True
            red_line = np.ones_like(f_filtered)
            ax.bar(f_filtered, red_line, bottom=current_sum + props, color="red", width=width)
            current_sum += red_line

        current_sum += props


    ax.set_xlabel(r"$\log_{10} (f \,/\, \text{Hz})$", fontsize = 22)
    ax.set_ylabel("Contribution [%]", fontsize = 22)
    ax.set_ylim(0, 100)

    if len(f_filtered) > 0:
        ax.set_xlim(f_filtered.min() - width, f_filtered.max() + width)

    sm = plt.cm.ScalarMappable(cmap=ccmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('Redshift ($z$)')

    plt.tight_layout()
    if save:
        label = metallicity if isinstance(metallicity, str) else "all_z"
        fig.savefig(
            f"../output/Figures/z_contr_{n_freq}_{n_int}_SFH{sfh}{sfh_type}_"
            f"{tag}_{label}_{compact_object}{subfolder}.png"
        )

    plt.show()

# --- System-count summaries and distributions ---

def distr_over_f(
    log_f_low: int,
    log_f_high: int,
    sfh: int,
    sfh_type: str,
    n_freq: int,
    n_int: int,
    tag: str,
    metallicity: Union[str, list[str]],
    compact_object: str,
    save_fig: bool = False,
    subfolder: str = "",
) -> None:
    """!
    @brief Plot system counts across frequency bins for bulk, birth, merger, and total.
    @param log_f_low: Base-10 logarithm of the lower frequency bound in Hz.
    @param log_f_high: Base-10 logarithm of the upper frequency bound in Hz.
    @param sfh: Numeric star-formation-history selector.
    @param sfh_type: Tabulated history label, if used.
    @param n_freq: Number of frequency bins.
    @param n_int: Number of integration bins.
    @param tag: Output run tag.
    @param metallicity: One metallicity key or sequence of keys to combine.
    @param compact_object: Population type label, such as WD or BH.
    @param save_fig: Save the figure when true.
    @param subfolder: Output subfolder or legacy dataset selector.
    @details Displays the chart and optionally saves it under output/Figures.
    """
    d1, d2, d3 = load_data(
        sfh, sfh_type, n_freq, n_int, tag, metallicity, compact_object, subfolder
    )

    if d1 is None: return

    totals1, totals2, totals3 = [], [], []
    for n in range(n_freq):
        totals1.append(np.sum(d1[f"freq_{n}_num"]))
        totals2.append(np.sum(d2[f"freq_{n}_num"]))
        totals3.append(np.sum(d3[f"freq_{n}_num"]))

    grand_total = np.array(totals1) + np.array(totals2) + np.array(totals3)

    f_range = np.logspace(log_f_low, log_f_high, 2*n_freq+1, base=10)
    f_plot = np.log10(np.array([f_range[2*i+1] for i in range(n_freq)]))
    width = (f_plot[1] - f_plot[0]) if len(f_plot) > 1 else 0.1

    fig, ax = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    titles = ["Bulk", "Birth", "Merge", "Total"]
    datasets = [totals1, totals2, totals3, grand_total]
    colors = ["red", "red", "red", "black"]

    for i in range(4):
        ax[i].bar(f_plot, datasets[i], color=colors[i], width=0.8*width, alpha=0.7)
        ax[i].set_title(titles[i])
        ax[i].grid(color="gainsboro", alpha=0.7)
        ax[i].set_yscale('log')
        ax[i].set_ylim(10**5, 1e21)
        ax[i].set_xlabel(r"$\log_{10} (f \,/$ Hz)")

    ax[0].set_ylabel("Number of systems")

    if save_fig:
        label = "TOTAL" if not isinstance(metallicity, str) else metallicity
        fig.savefig(
            f"../output/Figures/distr_f_{n_freq}_{n_int}_SFH{sfh}{sfh_type}_"
            f"{tag}_{label}_{compact_object}{subfolder}.png"
        )

    plt.show()

def distr_over_z(
    sfh: int,
    sfh_type: str,
    n_freq: int,
    n_int: int,
    tag: str,
    metallicity: Union[str, list[str]],
    compact_object: str,
    save_fig: bool = False,
    subfolder: str = "",
) -> None:
    """!
    @brief Create a bar chart of the number of systems in the different redshift bins.
    @param sfh: Numeric star-formation-history selector.
    @param sfh_type: Tabulated history label, if used.
    @param n_freq: Number of frequency bins represented by the data.
    @param n_int: Number of integration bins represented by the data.
    @param tag: Output run tag.
    @param metallicity: One metallicity key or sequence of keys to combine.
    @param compact_object: Population type label, such as WD or BH.
    @param save_fig: Save the figure when true.
    @param subfolder: Output subfolder or legacy dataset selector.
    @details Displays the chart and optionally saves it under output/Figures.
    """
    d1, d2, d3 = load_data(
        sfh, sfh_type, n_freq, n_int, tag, metallicity, compact_object, subfolder
    )

    if d1 is None: return

    num_cols = [c for c in d1.columns if "freq" in c and "num" in c]

    totals1 = d1[num_cols].sum(axis=1).values
    totals2 = d2[num_cols].sum(axis=1).values
    totals3 = d3[num_cols].sum(axis=1).values
    grand_total = totals1 + totals2 + totals3

    z_plot = d1["z"].values
    width = (z_plot[1] - z_plot[0]) if len(z_plot) > 1 else 0.1

    fig, ax = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    titles = ["Bulk", "Birth", "Merge", "Total"]
    datasets = [totals1, totals2, totals3, grand_total]
    colors = ["red", "red", "red", "black"]

    for i in range(4):
        ax[i].bar(z_plot, datasets[i], color=colors[i], width=0.8*width, alpha=0.7)
        ax[i].set_title(titles[i])
        ax[i].grid(color="gainsboro", alpha=0.7)
        ax[i].set_yscale('log')
        ax[i].set_ylim(10**5, 1e21)
        ax[i].set_xlabel(r"$z$")

    ax[0].set_ylabel("Number of systems")

    if save_fig:
        label = "TOTAL" if not isinstance(metallicity, str) else metallicity
        fig.savefig(
            f"../output/Figures/distr_z_{n_freq}_{n_int}_SFH{sfh}{sfh_type}_"
            f"{tag}_{label}_{compact_object}{subfolder}.png"
        )

    plt.show()


def summary(
    sfh: int,
    sfh_type: str,
    n_freq: int,
    n_int: int,
    tag: str,
    metallicity: Union[str, list[str]],
    compact_object: str,
    subfolder: str = "",
) -> None:
    """!
    @brief Print a summary of the number of systems in the different parts of the code.
    @param sfh: The star formation history.
    @param sfh_type: Tabulated history label, if used.
    @param n_freq: The number of frequency bins.
    @param n_int: The number of integration bins.
    @param tag: The tag of the output files.
    @param metallicity: Single string (e.g., "z02") or list of strings.
    @param compact_object: Compact-object population type, such as "BH" or "WD".
    @param subfolder: The subfolder of the output files.
    @return None. Prints system-count totals to standard output.
    """

    d1, d2, d3 = load_data(
        sfh, sfh_type, n_freq, n_int, tag, metallicity, compact_object, subfolder
    )

    if d1 is None: return

    totals1b = []
    totals2b = []
    totals3b = []

    for n in range(n_freq):
        totals1b.append(np.sum(d1[f"freq_{n}_num"]))
        totals2b.append(np.sum(d2[f"freq_{n}_num"]))
        totals3b.append(np.sum(d3[f"freq_{n}_num"]))

    label = metallicity if isinstance(metallicity, str) else "all_z"

    print(f"\n{tag}, {n_freq} bins, {label}, {compact_object}:\n")

    print(f"Total number of systems in bulk: {np.sum(np.array(totals1b)):.5E}")
    print(f"Total number of systems in birth bins: {np.sum(np.array(totals2b)):.5E}")
    print(f"Total number of systems in merge bins: {np.sum(np.array(totals3b)):.5E}")
    non_bulk_total = np.sum(totals2b) + np.sum(totals3b)
    print(f"Total number of systems in non-bulk bins: {non_bulk_total:.5E}")

    print(f"\nTOTAL: {np.sum(totals1b) + np.sum(totals2b) + np.sum(totals3b):.5E}")

def main() -> None:
    """! @brief Read post-processing settings and generate requested figures and summaries.
    @details Takes the parameter-file path as its first command-line argument.
    """
    if len(sys.argv) < 2:
        print(
            "Please provide a parameter file as "
            "'python Data_visualisation.py <parameter_file>'"
        )
        sys.exit(1)
    param_file = sys.argv[1]

    # Resolve paths relative to the src directory, as in the other scripts.
    if os.getcwd().split("/")[-1] != "src":
        print("Changing directory to src.")
        os.chdir("src")
        param_file = "../" + param_file
        param_file = Path(param_file)
        print(f"Changed parameter file path to {param_file}")

    config = cfg.ConfigParser()
    config.read(param_file)

    log_f_low = config.getfloat("integration", "log_f_low", fallback=-5)
    log_f_high = config.getfloat("integration", "log_f_high", fallback=4)
    n_freq = config.getint("integration", "N_freq", fallback=50)
    n_int = config.getint("integration", "N_int", fallback=20)
    max_z = config.getfloat("integration", "max_z", fallback=8)

    sfh_num = config.getint("physics", "SFH_num", fallback=1)
    sfh_type = config.get("physics", "SFH_type", fallback="")
    loop = config.getboolean("physics", "loop_over_metallicity", fallback=True)
    compact_object = config.get("physics", "object", fallback="BH")

    tag = config.get("settings", "tag", fallback="")
    integration_mode = config.get("settings", "integration_mode", fallback="")

    compare = config.getboolean("postproc", "compare", fallback=True)
    save = config.getboolean("postproc", "save_pp_figures", fallback=True)
    z_raw = config.get("postproc", "zfig", fallback="")
    z_list = z_raw.replace(" ", "").split(",") if z_raw else None
    subfolder = config.get("postproc", "subfolder", fallback="")
    if subfolder == "SN24":
        n_freq = 50
        n_int = 20
        max_z = 8
        sfh_num = 1
        compact_object = "WD"
        loop = False
        compare = False
    if subfolder == "HN24":
        n_freq = 50
        n_int = 20
        max_z = 8
        sfh_num = 6
        compact_object = "WD"
        integration_mode = "time"
        loop = True
        compare = False

    if z_list:
        multiplot(
            sfh_num,
            sfh_type,
            n_freq,
            n_int,
            tag,
            compact_object,
            save,
            "chosen_z",
            compare,
            z_list,
            subfolder=subfolder,
        )

    if loop:
        metallicities = ["z0001", "z0005", "z001", "z005", "z01", "z02", "z03"]
        multiplot(
            sfh_num,
            sfh_type,
            n_freq,
            n_int,
            tag,
            compact_object,
            save,
            "all_z",
            compare,
            subfolder=subfolder,
        )

        if sfh_num == 6:
            print("SFH6: Generating combined charts for all metallicities...")
            stacked_bar_chart(
                log_f_low, log_f_high, sfh_num, sfh_type, n_freq, n_int, tag,
                metallicities, compact_object, integration_mode, max_z, save,
                subfolder=subfolder,
            )
            distr_over_f(
                log_f_low, log_f_high, sfh_num, sfh_type, n_freq, n_int, tag,
                metallicities, compact_object, save, subfolder,
            )
            distr_over_z(
                sfh_num, sfh_type, n_freq, n_int, tag, metallicities,
                compact_object, save, subfolder,
            )
            summary(
                sfh_num, sfh_type, n_freq, n_int, tag, metallicities,
                compact_object, subfolder,
            )

        else:
            for z in metallicities:
                print(f"Generating individual charts for {z}...")
                stacked_bar_chart(
                    log_f_low, log_f_high, sfh_num, sfh_type, n_freq, n_int, tag,
                    z, compact_object, integration_mode, max_z, save,
                    subfolder=subfolder,
                )
                distr_over_f(
                    log_f_low, log_f_high, sfh_num, sfh_type, n_freq, n_int, tag,
                    z, compact_object, save, subfolder,
                )
                distr_over_z(
                    sfh_num, sfh_type, n_freq, n_int, tag, z, compact_object,
                    save, subfolder,
                )
                summary(
                    sfh_num, sfh_type, n_freq, n_int, tag, z, compact_object,
                    subfolder,
                )
    else:
        metallicity = config.get("physics", "metallicity", fallback="z02")
        multiplot(
            sfh_num,
            sfh_type,
            n_freq,
            n_int,
            tag,
            compact_object,
            save,
            "1z",
            compare,
            [metallicity],
            subfolder,
        )

        print(f"Generating charts for {metallicity}...")
        stacked_bar_chart(
            log_f_low, log_f_high, sfh_num, sfh_type, n_freq, n_int, tag,
            metallicity, compact_object, integration_mode, max_z, save,
            subfolder=subfolder,
        )
        distr_over_f(
            log_f_low, log_f_high, sfh_num, sfh_type, n_freq, n_int, tag,
            metallicity, compact_object, save, subfolder,
        )
        distr_over_z(
            sfh_num, sfh_type, n_freq, n_int, tag, metallicity,
            compact_object, save, subfolder,
        )
        summary(
            sfh_num, sfh_type, n_freq, n_int, tag, metallicity,
            compact_object, subfolder,
        )

if __name__ == "__main__":
    main()
