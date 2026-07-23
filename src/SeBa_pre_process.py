"""!
@file SeBa_pre_process.py
@date 2024-07-29
@author Seppe Staelens
@brief This program takes the output of the SeBa population synthesis code and calculates other values from it. The results are saved in a dataframe that can be used in the main code.
"""

import numpy as np
import sys
import os
from pathlib import Path
import pandas as pd
import configparser as cfg
from pathlib import Path

from modules.physics import *
from modules.auxiliary import tau_syst

if len(sys.argv) < 2:
    print("Please provide a parameter file as 'python SeBa_pre_process.py <parameter_file>'")
    sys.exit(1)

param_file = sys.argv[1]

# change directory to src
if os.getcwd().split("/")[-1] != "src":
    print("Changing directory to src.")
    os.chdir("src")
    param_file = "../" + param_file
    param_file = Path(param_file)
    print(f"Changed parameter file path to {param_file}")

config = cfg.ConfigParser()
config.read(param_file)

def pre_proc(metallicity: str) -> None:
    """!
    @brief Function to calculate values from the SeBa output and save them in a dataframe.
    """

    # --- Settings --- #
    SAVE_FILE = config.getboolean("settings", "save_datafile", fallback=False)
    object = config.get("physics", "object", fallback="WD")
    pop_synth = config.get("physics", "pop_synth", fallback="AlphaAlpha")
    alpha = config.get("physics", "alpha", fallback = "Alpha1")
    # add _Seppe before .dat.gz to run his data
    if object == "BH":
        data_file = Path(f"../data/{object}/{pop_synth}/{alpha}/{metallicity}/{metallicity}_t0aim1m1.dat")
    if object == "WD":
        data_file = Path(f"../data/{object}/{pop_synth}/{alpha}/{metallicity}/{metallicity}_t0aim1m1.dat.gz")

    if not data_file.exists():
        print(f"File {data_file} does not exist. Skipping.")
        return
    population = pd.read_csv(data_file, names=["t0", "a", "m1", "m2"], sep="\s+")

    # Calculate the initial orbital frequency from a_i, based on Kepler's law.
    # The GW frequency is twice the orbital frequency.
    m1 = population.m1
    m2 = population.m2
    orbital_frequencies = 1 / Period(population.a, m1, m2)
    population["nu0"] = orbital_frequencies

    # Calculate the chirp masses
    population["M_ch"] = chirp(m1, m2)

    # Calculate the factor K
    population["K"] = K(population.M_ch)

    # Calculate the maximal frequencies. The loop is because of a boolean operation in Kepler
    nu_list = []
    if object == "BH":
        for ma,mb in zip(m1, m2):
            nu_list.append(nu_max_BH(ma,mb))
    if object == "WD":
        for ma, mb in zip(m1, m2):
            nu_list.append(Kepler(ma, mb))
    population["nu_max"] = np.array(nu_list)

    # Calculate the maximal time to coalescence
    population["Dt_max"] = tau_syst(
        2 * population.nu0, 2 * population.nu_max, population.K
    )
    save_filename = Path(
        f"../data/{object}/{pop_synth}/{alpha}/{metallicity}/Initials_{metallicity}.txt.gz"
    )  # add _Seppe before .dat.gz when running his data

    if SAVE_FILE:
        population.to_csv(save_filename, index=False)


def main() -> None:
    """!
    @brief Main function.
    @details The main function checks whether the metallicity is looped over and runs the pre-processing.
    """

    loop = config.getboolean("physics", "loop_over_metallicity", fallback=False)
    if loop:
        metallicities = ["z0001", "z0005", "z001", "z005", "z01", "z02", "z03"]
        for i, metallicity in enumerate(metallicities):
            pre_proc(metallicity=metallicity)
    else:
        metallicity = config.get("physics", "metallicity", fallback="z02")
        pre_proc(metallicity=metallicity)


main()
