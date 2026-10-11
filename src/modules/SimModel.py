"""!
@package SimModel
@brief This module contains the class SimModel.
@details The class SimModel contains information about the run that needs to be shared over the different subroutines.
@author Seppe Staelens
@date 2024-07-24
"""

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import configparser as cfg
from modules.auxiliary import get_bin_factors, get_width_z_shell_from_z
import modules.RedshiftInterpolator as ri
import modules.SFRInterpolator as sfri
import numpy as np
from pathlib import Path


class SimModel:
    """!
    @brief Hold shared configuration and derived bin/cosmology values for a run.
    """

    ## lower bound of the frequency bins in log10 space.
    log_f_low: float
    ## upper bound of the frequency bins in log10 space.
    log_f_high: float
    ## number of frequency bins
    n_freq: int
    ## number of integration bins (z or T)
    n_int: int
    ## max_redshift
    max_z: float
    ## which star formation history to select. 1: Madau & Dickinson 2014, 2-4: made up, 5: constant 0.01.
    sfh_num: int
    ## type of star formation history. Can be 'MZ19', 'LZ19', 'HZ19', 'LZ21', 'HZ21' or 'MD' (Madau & Dickinson). Only used in case sfh_num = 6
    sfh_type: str
    ## metallicity, can be 'z0001', 'z001', 'z005', 'z01', 'z02' or 'z03'
    metallicity: str
    ## population synthesis model, can be 'AlphaAlpha' or 'GammaAlpha'
    pop_synth: str
    ## numerical value for alpha, can be 'Alpha1' or 'Alpha4'
    alpha: str
    ## normalisation for the population synthesis files in solar masses. 3.4e6 for Sophie, 4e6 for Seppe.
    normalisation: float
    ## population file
    population_file_name: str
    ## redshift interpolator file
    ri_file: str
    ## tag for filenames
    tag: str
    ## Integrate over "redshift" or (cosmic) "time"
    integ_mode: str
    ## Run script with(out) saving figures
    save_fig: bool
    ## Run script with(out) more output
    debug: bool
    ## Run script for only one system if True
    test_for_one: bool

    ## Redshift interpolator used for quick conversions in the cosmology
    z_interp: ri.RedshiftInterpolator
    ## Star Formation Rate interpolator to determine the sfr at a given redshift
    sfr_interp: sfri.SFRInterpolator

    ## The speed of light in units of Mpc/Myr
    light_speed = 0.30660139
    ## lookback time to maximal redshift in Myr
    max_lookback_time: float
    ## prefactor for the bulk calculations. value = 2.4e-15, value = 2e-15 for Seppe
    omega_prefactor_bulk: float
    ## prefactor for the birth and merger calculations. value = 3.76e-15, value = 3.2e-15 for Seppe
    omega_prefactor_birth_merger: float

    def __init__(self, input_file: str, metallicity: str) -> None:
        """!
        Initializes the SimModel by reading parameters, creating interpolators,
        and calculating bins and cosmology.
        @param input_file: Path to the INI parameter file.
        @param metallicity: Population metallicity key, such as "z02".
        """
        self.metallicity = metallicity
        self.read_params(input_file)

        self.z_interp = ri.RedshiftInterpolator(self.ri_file)
        self.sfr_interp = sfri.SFRInterpolator(
            self.z_interp, self.sfh_num, self.sfh_type, self.metallicity, self.max_z
        )

        self.calculate_f_bins()

        if self.integ_mode == "redshift":
            self.calculate_z_bins()
            self.calculate_cosmology_from_z()
        elif self.integ_mode == "time":
            self.calculate_t_bins()
            self.calculate_cosmology_from_time(self.z_interp)

    def read_params(self, input_file: str) -> None:
        """!
        @brief Read run settings from an INI parameter file.
        @param input_file: Path to the INI parameter file.
        @details Populates configuration and file paths from the integration,
        physics, files, and settings sections.
        """
        config = cfg.ConfigParser()
        config.read(input_file)

        self.log_f_low = config.getfloat("integration", "log_f_low", fallback=-5)
        self.log_f_high = config.getfloat("integration", "log_f_high", fallback=0)
        assert (
            self.log_f_low < self.log_f_high
        ), "log_f_low should be smaller than log_f_high"
        self.n_freq = config.getint("integration", "N_freq", fallback=50)
        self.n_int = config.getint("integration", "N_int", fallback=20)
        self.max_z = config.getfloat("integration", "max_z", fallback=8)

        self.sfh_num = config.getint("physics", "SFH_num", fallback=1)
        if self.sfh_num != 6:
            print("SFH_num is not 6, so sfh_type and metallicity are ignored.")
        self.sfh_type = config.get("physics", "SFH_type", fallback="")
        self.compact_object = config.get("physics", "object", fallback="WD")
        self.pop_synth = config.get("physics", "pop_synth", fallback="GammaAlpha")
        self.alpha = config.get("physics", "alpha", fallback="Alpha4")
        if self.compact_object == "BH":
            self.normalisation = 5.4e7
        if self.compact_object == "WD":
            self.normalisation = 3.4e6 #change to 4e6 for _seppe data
        self.omega_prefactor_bulk = 8.10e-9 / self.normalisation
        self.omega_prefactor_birth_merger = 1.28e-8 / self.normalisation

        self.population_file_name = (
            "../data/"
            + self.compact_object
            + "/"
            + self.pop_synth
            + "/"
            + self.alpha
            + "/"
            + self.metallicity
            + "/"
            + "Initials_"
            + self.metallicity
        )
        if config.getboolean("files", "use_data_Seppe", fallback=False):
            self.population_file_name += "_Seppe"
        self.population_file_name += ".txt.gz"
        self.population_file_name = Path(self.population_file_name)
        self.ri_file = Path(
            "../data/" + config.get("files", "ri_file", fallback="z_at_age.txt")
        )

        self.tag = config.get("settings", "tag", fallback="")
        self.integ_mode = config.get(
            "settings", "integration_mode", fallback="redshift"
        )
        self.output_path = config.get(
            "settings", "output_path", fallback="../output/GWBs/"
        )
        self.save_fig = config.getboolean("settings", "save_fig", fallback=False)
        self.debug = config.getboolean("settings", "debug", fallback=False)
        self.test_for_one = config.getboolean(
            "settings", "test_for_one", fallback=False
        )

    def calculate_f_bins(self) -> None:
        """!
        @brief Calculate frequency sample points, bin edges, and bin factors.
        @details Populates f_plot, f_bins, and f_bin_factors.
        """
        f_range = np.logspace(
            self.log_f_low, self.log_f_high, 2 * self.n_freq + 1, base=10
        )
        ## The frequencies at which we will plot
        self.f_plot = np.array([f_range[2 * i + 1] for i in range(self.n_freq)])
        ## The frequency bins
        self.f_bins = np.array([f_range[2 * i] for i in range(self.n_freq + 1)])
        ## The frequency bin factors that appear in the calculation
        self.f_bin_factors = get_bin_factors(self.f_plot, self.f_bins)

        print(f"\nThe frequencies are {self.f_plot}\n")

    def calculate_z_bins(self) -> None:
        """!
        @brief Calculate uniform redshift sample points and bin edges.
        @details Populates z_list and z_bins from zero to max_z.
        """
        z_range = np.linspace(0, self.max_z, 2 * self.n_int + 1)
        ## The central values of the redshift bins
        self.z_list = np.array([z_range[2 * i + 1] for i in range(self.n_int)])
        ## The redshift bins
        self.z_bins = np.array([z_range[2 * i] for i in range(self.n_int + 1)])

        print(f"The redshifts are {self.z_list}\n")

    def calculate_t_bins(self) -> None:
        """!
        @brief Calculate uniform lookback-time sample points and bin edges.
        @details Populates max_lookback_time, t_range, t_list, t_bins, and dt; times are in Myr.
        """
        self.max_lookback_time = cosmo.lookback_time(self.max_z).to(u.Myr)

        self.t_range = np.linspace(0, self.max_lookback_time.value, 2 * self.n_int + 1)

        self.t_list = np.array([self.t_range[2 * i + 1] for i in range(self.n_int)])
        self.t_bins = np.array([self.t_range[2 * i] for i in range(self.n_int + 1)])

        self.dt = self.t_list[1] - self.t_list[0]

        print(f"The cosmic timestep is {self.dt} Myr\n")
        print(f"The times are {self.t_list}\n")

    def calculate_cosmology_from_z(self) -> None:
        """!
        @brief Derive cosmological quantities from the redshift bins.
        @details Populates shell widths, elapsed times since max_z, Universe
        ages, and max_lookback_time using Planck18 cosmology.
        """
        ## The width of the redshift bins in Mpc
        self.z_widths = get_width_z_shell_from_z(self.z_bins)
        ## The time since the maximum redshift
        self.z_time_since_max_z = (
            cosmo.lookback_time(self.max_z) - cosmo.lookback_time(self.z_list)
        ).to(u.Myr)
        ## The age of the universe at each redshift
        self.ages = (cosmo.age(0) - cosmo.lookback_time(self.z_list)).to(u.Myr)

        self.max_lookback_time = cosmo.lookback_time(self.max_z).to(u.Myr)

    def calculate_cosmology_from_time(
        self, z_interpolator: ri.RedshiftInterpolator
    ) -> None:
        """!
        @brief Derive cosmological quantities from the lookback-time bins.
        @details Populates ages, redshifts, and elapsed times since max_z.
        @param z_interpolator: Lookup used to convert Universe ages to redshifts.
        """
        self.ages = cosmo.age(0).to(u.Myr) - self.t_list * u.Myr
        self.z_list = z_interpolator.get_z_fast(self.ages.value)
        self.z_time_since_max_z = (self.max_lookback_time.value - self.t_list) * u.Myr

        print(f"The redshifts are {self.z_list}\n")
