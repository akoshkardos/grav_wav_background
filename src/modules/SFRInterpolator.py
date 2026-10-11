"""!
@package SFRInterpolator
@brief This module contains the class SFRInterpolator.
@details The class selects an analytic or tabulated star formation rate history
and evaluates it at a redshift or at a Universe age converted to redshift.
@author Seppe Staelens
@date 2024-09-01
"""

from numpy import interp
import pandas as pd
from pathlib import Path
import modules.SFH as sfh
import modules.RedshiftInterpolator as ri


class SFRInterpolator:
    """!
    @brief Interpolate or evaluate a selected star formation rate history.
    @details sfh numbers 1-5 select analytic histories; number 6 loads a
    tabulated history selected by sfh_type and metallicity. Rates are in
    solar masses per year per cubic megaparsec.
    """

    def __init__(
        self,
        redshift_interpolator: ri.RedshiftInterpolator,
        sfh_num: int = 1,
        sfh_type: str = "MZ19",
        metallicity: str = "z02",
        max_z: float = 8.0,
    ) -> None:
        """!
        Initializes the SFRInterpolator object.
        @param redshift_interpolator: Converts Universe age to redshift.
        @param sfh_num: History selector: 1 is Madau-Dickinson (2014), 2-4 are
        analytic alternatives, 5 is constant, and 6 selects tabulated data.
        @param sfh_type: Tabulated history name (LZ19, MZ19, HZ19, LZ21, HZ21);
        used only when sfh_num is 6.
        @param metallicity: Tabulated metallicity key (z03, z02, z01, z005,
        z001, or z0001); used only when sfh_num is 6.
        @param max_z: Maximum redshift used by representative_sfh, dimensionless.
        """

        self.redshift_interpolator = redshift_interpolator
        self.max_z = max_z

        if sfh_num == 1:

            def sfr_impl(z: float) -> float:
                return sfh.sfh_md(z)

        elif sfh_num == 2:

            def sfr_impl(z: float) -> float:
                return sfh.sfh2(z)

        elif sfh_num == 3:

            def sfr_impl(z: float) -> float:
                return sfh.sfh3(z)

        elif sfh_num == 4:

            def sfr_impl(z: float) -> float:
                return sfh.sfh4(z)

        elif sfh_num == 5:

            def sfr_impl(z: float) -> float:
                return 0.01

        elif sfh_num == 6:
            sfr_at_val_data = pd.read_csv(
                Path(f"../data/SFRD/{sfh_type}_SFRD_allbins.txt.gz")
            )
            # the order is reversed to be in ascending order, compatible with numpy.interp
            self.interp_z = sfr_at_val_data.redshift.values[::-1]
            # Number in square brackets corresponds to different metallicities
            if metallicity == "z03":
                self.interp_sfr = sfr_at_val_data["0"].values[::-1]
            elif metallicity == "z02":
                self.interp_sfr = sfr_at_val_data["1"].values[::-1]
            elif metallicity == "z01":
                self.interp_sfr = sfr_at_val_data["2"].values[::-1]
            elif metallicity == "z005":
                self.interp_sfr = sfr_at_val_data["3"].values[::-1]
            elif metallicity == "z001":
                self.interp_sfr = sfr_at_val_data["4"].values[::-1]
            elif metallicity == "z0001":
                self.interp_sfr = sfr_at_val_data["5"].values[::-1]
            else:
                raise ValueError(
                    "Invalid metallicity value. Choose from 'z03', 'z02', 'z01', "
                    "'z005', 'z001', or 'z0001'."
                )

            def sfr_impl(z: float) -> float:
                return interp(z, self.interp_z, self.interp_sfr)

        else:
            raise ValueError("Invalid sfh_num value. Choose from 1, 2, 3, 4, 5 or 6.")

        self._sfr_impl = sfr_impl

    def sfr(self, z: float) -> float:
        """!
        @brief Evaluate the selected star formation rate at a redshift.
        @param z: Redshift (dimensionless).
        @return Star formation rate in solar masses per year per cubic megaparsec.
        """
        return self._sfr_impl(z)

    def representative_sfh(self, age: float, delta_t: float = 0.0) -> float:
        """!
        @brief Determine the selected star formation rate at a Universe age.
        @details Subtracts the optional delay before converting the resulting age
        to redshift with the configured RedshiftInterpolator.
        @param age: Universe age in Myr.
        @param delta_t: Formation or frequency-bin delay in Myr; defaults to zero.
        @return Star formation rate in solar masses per year per cubic megaparsec.
        """
        new_age = age - delta_t
        z_new = self.redshift_interpolator.get_z_fast(new_age)
        if z_new > self.max_z:
            print(f"z larger than {self.max_z}")

        return self.sfr(z_new)
