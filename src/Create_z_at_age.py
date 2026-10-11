"""!
@file Create_z_at_age.py
@date 2024-07-29
@author Seppe Staelens
@brief This program creates a list of redshift values at a list of ages of the Universe, that can be saved and used to interpolate in the main code.
"""

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import os
from pathlib import Path

print("Current dir: " + os.getcwd())


def main() -> None:
    """!
    @brief Create a redshift lookup table indexed by the age of the Universe.
    @details Uses Planck18 cosmology and writes the table to ../data/z_at_age.txt.
    The maximum redshift and number of interpolation points are set in the
    settings block below.
    """

    # ----- SETTINGS ----- #
    max_z = 8  # Maximum redshift
    nr_interp = 10000  # Number of points at which to calculate the redshift

    # ----- CALCULATIONS ----- #
    initial_age = cosmo.age(max_z).to(u.Myr)
    current_age = cosmo.age(1e-5).to(u.Myr)

    ages = np.linspace(initial_age.value, current_age.value, nr_interp)
    z_vals = np.array(z_at_value(cosmo.age, ages * u.Myr).value)

    data = pd.DataFrame({"Age (Myr)": ages, "Redshift": z_vals})
    data.to_csv(Path("../data/z_at_age.txt"), index=False)


main()
