"""!
@file SFH.py
@date 2024-07-24
@brief This file contains analytic functions to determine the star formation rate.
@details The file contains analytic functions to determine the star formation rate. 
The functions sfh_md and sfh2 through sfh4 are analytic histories selectable
by SFRInterpolator. Other histories are loaded from data files.
@author Seppe Staelens
"""


def sfh_md(z: float) -> float:
    """!
    @brief Star formation history from [Madau, Dickinson 2014].
    @param z: redshift.
    @return sfr: star formation rate. Units: solar mass / yr / Mpc^3.
    """
    return 0.015 * (1 + z) ** (2.7) / (1 + ((1 + z) / 2.9) ** (5.6))


def sfh2(z: float) -> float:
    """!
    @brief Made up star formation history.
    @param z: redshift.
    @return sfr: star formation rate. Units: solar mass / yr / Mpc^3.
    """
    return 0.143 * (1 + z) ** (0.3) / (1 + ((1 + z) / 2.9) ** (3.2))


def sfh3(z: float) -> float:
    """!
    @brief Made up star formation history.
    @param z: redshift.
    @return sfr: star formation rate. Units: solar mass / yr / Mpc^3.
    """
    return 0.00533 * (1 + z) ** (2.7) / (1 + ((1 + z) / 2.9) ** (3.0))


def sfh4(z: float) -> float:
    """!
    @brief Made up star formation history.
    @param z: redshift.
    @return sfr: star formation rate. Units: solar mass / yr / Mpc^3.
    """
    return 0.00245 * (1 + z) ** (2.7) / (1 + ((1 + z) / 5.0) ** (5.6))
