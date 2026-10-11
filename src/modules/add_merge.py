"""!
@file add_merge.py
@brief This file contains a routine that adds the contribution of the 'merger bins' due to kepler max to the bulk+birth GWB.
@author Seppe Staelens
@date 2024-07-24
"""

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from modules.auxiliary import make_omega_plot_unnorm, tau_syst, determine_upper_freq
import modules.SimModel as sm
from pathlib import Path


def add_merge(model: sm.SimModel, data: pd.DataFrame) -> None:
    """!
    @brief Add the merger-bin contribution to the bulk and birth GWB.
    @param model: instance of SimModel, containing the necessary information for the run.
    @param data: dataframe containing the binary population data.
    @details Reads the previously saved bulk-plus-birth spectrum, then writes
    the updated spectrum and merger-bin contribution/count table to
    model.output_path. Integration mode is selected by model.integ_mode;
    model.save_fig controls whether a spectrum plot is saved.
    """

    print("\nInitiating merger bin part of the code.\n")

    previous_Omega = pd.read_csv(
        Path(
            model.output_path
            + f"SFH{model.sfh_num}{model.sfh_type}_{model.n_freq}_{model.n_int}_"
            f"{model.metallicity}_wbirth_{model.tag}_{model.compact_object}.txt"
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

    # We will have no merger bin for binaries that have f_max above our region of interest
    highest_bin = model.f_bins[-1]
    lowest_bin = model.f_bins[0]

    # To check numerics
    num_errors = 0

    # We go over the rows in the data and determine the merger bins, and their contribution
    for index, row in data.iterrows():

        if model.test_for_one and (index > 0):
            break

        if index % 1000 == 0:
            print(f"At row {index} out of {len(data)}.")

        MERGER_CAN_BE_REACHED = True

        # Determine merger bins for every z bin
        for i, z in enumerate(model.z_list):

            # Don't consider mergers that happen at a frequency beyond our region of interest
            if 2 * row.nu_max / (1 + z) > highest_bin:
                continue

            evolve_time = model.z_time_since_max_z[i].value - row.t0

            if evolve_time <= 0:
                continue

            if MERGER_CAN_BE_REACHED:

                # find merger bin
                bin_index = np.digitize(2 * row.nu_max / (1 + z), model.f_bins) - 1
                low_f_r, upp_f_r = model.f_bins[bin_index], model.f_bins[bin_index + 1]

                # The time it takes to evolve from birth to merger
                tau = row.Dt_max

                if tau >= evolve_time:
                    MERGER_CAN_BE_REACHED = False
                    if model.test_for_one:
                        print("Did not reach merger.")

                else:
                    if model.test_for_one:
                        print("Reached merger.")

                    # calculate representative sfh at the time of formation
                    psi = model.sfr_interp.representative_sfh(
                        model.ages[i].value, delta_t=tau
                    )

                    # contributions
                    freq_fac = (
                        row.nu_max ** (2 / 3) - (low_f_r * (1 + z) / 2) ** (2 / 3)
                    ) / (upp_f_r - low_f_r)
                    num_syst = (
                        psi * tau_syst(low_f_r * (1 + z), 2 * row.nu_max, row.K) * 10**6
                    )  # tau is given in Myr, psi in ... /yr

            if not MERGER_CAN_BE_REACHED:

                nu_max_b_ini = determine_upper_freq(row.nu0, evolve_time, row.K)

                if nu_max_b_ini == -1:
                    # Should not be possible
                    num_errors += 1
                    continue

                if model.debug:
                    tolerance = 0.01
                    if nu_max_b_ini > (1 + tolerance) * row.nu_max:
                        # The first means that evolve_time too large, second should be caught in previous part
                        raise "Error"

                # for safety
                nu_max_b = min(nu_max_b_ini, row.nu_max)

                # Don't consider mergers that happen at a frequency beyond our region of interest
                if (2 * nu_max_b / (1 + z) > highest_bin) or (
                    2 * nu_max_b / (1 + z) < lowest_bin
                ):
                    continue

                # find merger bin
                bin_index = np.digitize(2 * nu_max_b / (1 + z), model.f_bins) - 1
                if bin_index == -1:
                    print("oops")

                low_f_r, upp_f_r = model.f_bins[bin_index], model.f_bins[bin_index + 1]

                if 2 * row.nu0 / (1 + z) > low_f_r:
                    # in this case the merger is already included as a special case in add birth contribution
                    continue

                freq_fac = (
                    nu_max_b ** (2 / 3) - (low_f_r * (1 + z) / 2) ** (2 / 3)
                ) / (upp_f_r - low_f_r)
                tau = tau_syst(2 * row.nu0, low_f_r * (1 + z), row.K)
                psi = model.sfr_interp.representative_sfh(
                    model.ages[i].value, delta_t=tau
                )

                num_syst = (
                    psi * (evolve_time - tau) * 10**6
                )  # tau is given in Myr, psi in ... /yr

                if model.debug:
                    np.testing.assert_allclose(
                        evolve_time,
                        tau + tau_syst(low_f_r * (1 + z), 2 * nu_max_b, row.K),
                        rtol=1e-2,
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

            if model.integ_mode == "redshift":
                z_contr.loc[i, f"freq_{bin_index}"] += omega_cont / (
                    model.omega_prefactor_bulk * model.f_bin_factors[bin_index]
                )
                z_contr.loc[i, f"freq_{bin_index}_num"]+= (
                    (4 * np.pi / model.normalisation)
                    * num_syst
                    * (cosmo.comoving_distance(z).value ** 2)
                    * model.z_widths[i]
                )
            elif model.integ_mode == "time":
                z_contr.loc[i, f"freq_{bin_index}"] += (
                    omega_cont / model.f_bin_factors[bin_index]
                )
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

    if model.debug:
        print(f"Number of numerical errors: {num_errors}\n")

    # Plots
    if model.save_fig:
        make_omega_plot_unnorm(
            model.f_plot,
            omega_plot,
            model.save_fig,
            f"GWB_SFH{model.sfh_num}{model.sfh_type}_{model.n_freq}_{model.n_int}_"
            f"wmerge_{model.tag}_{model.compact_object}"
        )

    # Save GWB
    GWBnew = pd.DataFrame({"f": model.f_plot, "Om": omega_plot})
    GWBnew.to_csv(
        Path(
            model.output_path
            + f"SFH{model.sfh_num}{model.sfh_type}_{model.n_freq}_{model.n_int}_"
            f"{model.metallicity}_wmerge_{model.tag}_{model.compact_object}.txt"
        ),
        index=False,
    )
    z_contr.to_csv(
        Path(
            model.output_path
            + f"SFH{model.sfh_num}{model.sfh_type}_{model.n_freq}_{model.n_int}_"
            f"{model.metallicity}_z_contr_merge_{model.tag}_"
            f"{model.compact_object}.txt"
        ),
        index=False,
    )
