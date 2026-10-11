"""!
@file add_bulk.py
@brief This file contains a routine that calculates the majority of the GWB, what is referred to in my thesis as the 'generic case'.
@author Seppe Staelens
@date 2024-07-24
"""

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from modules.auxiliary import make_omega_plot_unnorm, tau_syst
import modules.SimModel as sm
from pathlib import Path


def add_bulk(model: sm.SimModel, data: pd.DataFrame) -> None:
    """!
    @brief Calculate the bulk (generic-case) contribution to the GWB.
    @details Uses the model's configured integration mode (redshift or time).
    Writes the updated GWB spectrum and per-bin contribution/count tables to
    model.output_path. If model.save_fig is true, also saves a spectrum plot.
    @param model: Simulation configuration and derived integration quantities.
    @param data: Preprocessed binary population table.
    """

    print("\nInitiating bulk part of the code.\n")

    # array that will store the values for omega
    omega_plot = np.zeros_like(model.f_plot)

    # array that will store the contributions of each shell
    z_contr = pd.DataFrame({"z": model.z_list})
    if model.integ_mode == "time":
        z_contr["T"] = model.t_list

    # to stor the contributions for each frequency bin, and the number of systems
    for i in range(model.n_freq):
        z_contr[f"freq_{i}"] = np.zeros_like(model.z_list)
        z_contr[f"freq_{i}_num"] = np.zeros_like(model.z_list)

    # We now loop over the received frequency values f_r.
    for j, f_r in enumerate(model.f_plot):

        low_f_r, upp_f_r = model.f_bins[j], model.f_bins[j + 1]  # Bin edges
        omega = 0

        # We calculate the contribution to the frequency bin for every redshift bin
        for i, z in enumerate(model.z_list):
            time_since_max_z = model.z_time_since_max_z[i].value
            age = model.ages[i].value
            bin_low_f_e = low_f_r * (1 + z)  # Emission frequency bin edges
            bin_upp_f_e = upp_f_r * (1 + z)  #
            z_fac = 0
            num_syst = 0

            # We calculate the contribution for every type of binary in the Population Synthesis
            for index, row in data.iterrows():

                if model.test_for_one and (index > 0):
                    break

                # Working on generic case, so strictly f_0 <  low_f_e < high_f_e < f_max
                if 2 * row.nu0 > bin_low_f_e or 2 * row.nu_max < bin_upp_f_e:
                    continue

                tau = tau_syst(
                    2 * row.nu0, bin_upp_f_e, row.K
                )  # Time to evolve from WD binary formation to upper edge of bin
                time_since_ZAMS = tau + row.t0  # Both quantities are in Myr

                # Binary can't be older then the beginning of the Universe (with max_z ~ the beginning)
                if time_since_ZAMS >= time_since_max_z:
                    continue

                # calculate sfr at the time of formation
                psi = model.sfr_interp.representative_sfh(age, delta_t=time_since_ZAMS)

                # binary specific contribution to the stored quantities
                z_fac += psi * row.M_ch ** (5 / 3)
                num_syst += (
                    psi * tau_syst(bin_low_f_e, bin_upp_f_e, row.K) * 10**6
                )  # tau is given in Myr, psi in ... /yr

            # the contribution if we integrate over T
            omega_cont = z_fac * (1 + z) ** (-1 / 3)

            # if we integrate over z, we need to add another factor (1+z)^(-1) Delta z
            if model.integ_mode == "redshift":
                omega_cont *= model.z_widths[i] * (1 + z) ** (-1)

            omega += omega_cont
            z_contr.loc[i, f"freq_{j}"] = omega_cont

            # the contribution to the number of systems
            pre_num = (
                (4 * np.pi / model.normalisation)
                * num_syst
                * (cosmo.comoving_distance(z).value ** 2)
            )
            if model.integ_mode == "redshift":
                z_contr.loc[i, f"freq_{j}_num"] = pre_num * model.z_widths[i]
            elif model.integ_mode == "time":
                z_contr.loc[i, f"freq_{j}_num"] = (
                    pre_num * model.light_speed * (1 + z) * model.dt
                )

        # We store the value of omega for this frequency bin
        omega_plot[j] = model.omega_prefactor_bulk * omega * model.f_bin_factors[j]
        if model.integ_mode == "time":
            omega_plot[j] *= model.light_speed * model.dt

        print(f"At frequency {f_r:.5f}: {omega_plot[j]:.3E}.")

    # Plots
    if model.save_fig:
        make_omega_plot_unnorm(
            model.f_plot,
            omega_plot,
            model.save_fig,
            f"GWB_SFH{model.sfh_num}{model.sfh_type}_{model.n_freq}_{model.n_int}_"
            f"{model.tag}_{model.compact_object}"
        )

    # Save GWB
    GWB = pd.DataFrame({"f": model.f_plot, "Om": omega_plot})
    GWB.to_csv(
        Path(
            model.output_path
            + f"SFH{model.sfh_num}{model.sfh_type}_{model.n_freq}_{model.n_int}_"
            f"{model.metallicity}_{model.tag}_{model.compact_object}.txt"
        ),
        index=False,
    )
    z_contr.to_csv(
        Path(
            model.output_path
            + f"SFH{model.sfh_num}{model.sfh_type}_{model.n_freq}_{model.n_int}_"
            f"{model.metallicity}_z_contr_{model.tag}_{model.compact_object}.txt"
        ),
        index=False,
    )
