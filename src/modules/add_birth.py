"""!
@file add_birth.py
@brief This file contains a routine that adds the contribution of the 'birth bins' to the bulk GWB.
@author Seppe Staelens
@date 2024-07-24
"""

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from modules.auxiliary import make_omega_plot_unnorm, tau_syst, determine_upper_freq
import modules.SimModel as sm
from pathlib import Path


def add_birth(model: sm.SimModel, data: pd.DataFrame) -> None:
    """!
    @brief Add the birth-bin contribution to the bulk GWB.
    @param model: instance of SimModel, containing the necessary information for the run.
    @param data: dataframe containing the binary population data.
    @details Reads the previously saved bulk spectrum, then writes the updated
    spectrum and birth-bin contribution/count table to model.output_path.
    Integration mode is selected by model.integ_mode; model.save_fig controls
    whether a spectrum plot is saved.
    """

    print("\nInitating birth bin part of the code.\n")

    # read the bulk part of the GWB
    previous_Omega = pd.read_csv(
        Path(
            model.output_path
            + f"SFH{model.sfh_num}{model.sfh_type}_{model.n_freq}_{model.n_int}_"
            f"{model.metallicity}_{model.tag}_{model.compact_object}.txt"
        ),
        sep=",",
    )
    omega_plot = previous_Omega.Om.values.copy()

    # Create dataframe to store results
    z_contr = pd.DataFrame({"z": model.z_list})
    if model.integ_mode == "time":
        z_contr["T"] = model.t_list

    for i in range(model.n_freq):
        z_contr[f"freq_{i}"] = np.zeros_like(model.z_list)
        z_contr[f"freq_{i}_num"] = np.zeros_like(model.z_list)

    # We will have no birth bin for binaries that have f_0 below our region of interest
    lowest_bin = model.f_bins[0]

    # We go over the rows in the data and determine the birth bins, and their contribution
    for index, row in data.iterrows():

        if model.test_for_one and (index > 0):
            break

        if index % 1000 == 0:
            print(f"At row {index} out of {len(data)}.")

        # Determine birth bins for every z bin
        for i, z in enumerate(model.z_list):
            time_since_max_z = model.z_time_since_max_z[i].value

            # Binaries can't be older than the Universe
            if row.t0 >= time_since_max_z:
                continue

            # Birth frequency out of our region of interest
            if 2 * row.nu0 / (1 + z) < lowest_bin:
                continue

            # determine the birth bin
            bin_index = np.digitize(2 * row.nu0 / (1 + z), model.f_bins) - 1
            low_f_r, upp_f_r = model.f_bins[bin_index], model.f_bins[bin_index + 1]
            if model.test_for_one:
                print(f"Bin frequencies for z {z:.2f}: [{low_f_r:.2E}, {upp_f_r:.2E}]")

            age = model.ages[i].value
            # calculate representative sfh at the time of formation
            psi = model.sfr_interp.representative_sfh(age, delta_t=row.t0)

            # The time it would take the binary to evolve from nu_0 to the upper bin edge
            tau_to_bin_edge = tau_syst(2 * row.nu0, upp_f_r * (1 + z), row.K)

            # If this time is larger than the time the binary has had to evolve since max_z,
            # the latter duration is used.
            max_evolve_time = time_since_max_z - row.t0
            if tau_to_bin_edge >= max_evolve_time:
                tau_in_bin = max_evolve_time
                upp_freq = determine_upper_freq(row.nu0, max_evolve_time, row.K)
                freq_fac = (upp_freq ** (2 / 3) - row.nu0 ** (2 / 3)) / (
                    upp_f_r - low_f_r
                )
            else:
                tau_in_bin = tau_to_bin_edge
                freq_fac = ((upp_f_r * (1 + z) / 2) ** (2 / 3) - row.nu0 ** (2 / 3)) / (
                    upp_f_r - low_f_r
                )

            # contributions
            omega_cont = (
                model.f_plot[bin_index]
                * row.M_ch ** (5 / 3)
                * freq_fac
                * (1 + z) ** (-1)
                * psi
            )
            if model.integ_mode == "redshift":
                omega_cont *= (
                    model.omega_prefactor_birth_merger
                    * (1 + z) ** (-1)
                    * model.z_widths[i]
                )

            num_syst = psi * tau_in_bin * 10**6  # tau is given in Myr, psi in ... /yr

            if model.integ_mode == "redshift":
                z_contr.loc[i, f"freq_{bin_index}"] += omega_cont / (
                    model.omega_prefactor_bulk * model.f_bin_factors[bin_index]
                )  # The denominator is to keep the relative size wrt the bulk
                z_contr.loc[i, f"freq_{bin_index}_num"] += (
                    (4 * np.pi / model.normalisation)
                    * num_syst
                    * (cosmo.comoving_distance(z).value ** 2)
                    * model.z_widths[i]
                )
            elif model.integ_mode == "time":
                z_contr.loc[i, f"freq_{bin_index}"] += (
                    omega_cont / model.f_bin_factors[bin_index]
                )  # The denominator is to keep the relative size wrt the bulk
                z_contr.loc[i, f"freq_{bin_index}_num"] += (
                    (4 * np.pi / model.normalisation)
                    * num_syst
                    * (cosmo.comoving_distance(z).value ** 2)
                    * model.light_speed
                    * (1 + z)
                    * model.dt
                )

            if model.integ_mode == "time":
                omega_cont *= (
                    model.light_speed * model.omega_prefactor_birth_merger * model.dt
                )

            omega_plot[bin_index] += omega_cont

    # Plots
    if model.save_fig:
        make_omega_plot_unnorm(
            model.f_plot,
            omega_plot,
            model.save_fig,
            f"GWB_SFH{model.sfh_num}{model.sfh_type}_{model.n_freq}_{model.n_int}_"
            f"wbirth_{model.tag}_{model.compact_object}"
        )

    # Save GWB
    GWBnew = pd.DataFrame({"f": model.f_plot, "Om": omega_plot})
    GWBnew.to_csv(
        Path(
            model.output_path
            + f"SFH{model.sfh_num}{model.sfh_type}_{model.n_freq}_{model.n_int}_"
            f"{model.metallicity}_wbirth_{model.tag}_{model.compact_object}.txt"
        ),
        index=False,
    )
    z_contr.to_csv(
        Path(
            model.output_path
            + f"SFH{model.sfh_num}{model.sfh_type}_{model.n_freq}_{model.n_int}_"
            f"{model.metallicity}_z_contr_birth_{model.tag}_"
            f"{model.compact_object}.txt"
        ),
        index=False,
    )
