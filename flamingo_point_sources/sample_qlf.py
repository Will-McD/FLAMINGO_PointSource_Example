#!/bin/env python
"""
Sampling quasar bolometric luminosities from the Shen+2020 quasar luminosity function (QLF), for abundance matching
the quasars of a lightcone shell to its haloes, and the Shen+2020 bolometric corrections.

The QLF parameter files (Shen2020_local_polished.dat, Shen2020_local_free.dat) are in QLF_models/ at the top of the
repository.
"""
import os
import numpy as np
import unyt
from scipy import integrate
from tqdm import tqdm
from swiftlet.snapshot_units import apply_expected_units


# directory of the Shen2020 QLF parameter files, QLF_models/ at the top of the repository
QLF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "QLF_models")

# other names of the QLF parameter columns, e.g. in Shen2020_local_free.dat
QLF_COLUMN_ALIASES = {"phi_s": "log_phi_star", "L_s": "log_L_star"}

# Shen+2020 bolometric correction coefficients (c1, c2, k1, k2)
SHEN2020_COEFFS = {
    "soft": (5.712, 17.67, -0.026, 0.278),
    "hard": (4.073, 12.60, -0.026, 0.278),
}

def Shen_2020_bolometric_correation(Lbol, band="soft"):
    """
    Shen+2020 bolometric correction for the soft (0.5-2 keV) or hard (2-10 keV) X-ray band.

    Returns Lbol / L_band

    :param  Lbol:   bolometric luminosity
    :type   Lbol:   unyt.unyt_array
    :param  band:   X-ray band, 'soft' or 'hard'
    :type   band:   str
    """
    if band not in SHEN2020_COEFFS:
        raise ValueError(f"unknown band '{band}', use 'soft' or 'hard'")
    c1, c2, k1, k2 = SHEN2020_COEFFS[band]
    q = (Lbol/(1e10*unyt.Lsun)).to_value(unyt.dimensionless)
    return c1*q**k1 + c2*q**k2


class SampleQLF:
    """
    Sample bolometric luminosities of quasars from the Shen+2020 quasar luminosity function (QLF)
    """
    def __init__(self, filename, seed=44):
        """
        Load the QLF parameters.

        :param  filename:   QLF parameter file, a text file with named columns z, gamma1, gamma2, log_phi_star
                            [Mpc^-3 dex^-1] and log_L_star [Lsun] (or phi_s and L_s, see QLF_COLUMN_ALIASES).
                            Rows starting with # are skipped. A path, or the name of a file in QLF_DIR
        :type   filename:   str
        :param  seed:       random seed of the sampled luminosities, see shell_rng
        :type   seed:       int
        """
        self.seed = int(seed)
        self.rng = np.random.RandomState(seed) # continuing stream, used if per_shell_seed=False
        self.qlf_params = self.fetch_all_parameters(filename)
        self.z_qlf = None       # tabulated redshift of the selected QLF
        self.z_ind = None       # its row in qlf_params
        self.qlf_model = None   # its parameters

    @property
    def model_params(self):
        """
        Dictionary of the QLF parameters:
            (z, gamma1, gamma2, log_phi_star, log_L_star)
        """
        if self.qlf_model is None:
            raise ValueError("No QLF selected. Define the QLF's redshift with update_qlf_redshift.")
        return dict(self.qlf_model)

    @staticmethod
    def fetch_all_parameters(filename):
        """
        Load the QLF parameters at all tabulated redshifts. 

        :param  filename:   QLF parameter file, a path or the name of a file in QLF_DIR
        :type   filename:   str
        """
        path = filename if os.path.isfile(filename) else os.path.join(QLF_DIR, filename)
        param_dat = np.genfromtxt(path, names=True)
        params = {QLF_COLUMN_ALIASES.get(name, name): np.atleast_1d(param_dat[name]) for name in param_dat.dtype.names}
        missing = [k for k in ("z", "gamma1", "gamma2", "log_phi_star", "log_L_star") if k not in params]
        if missing:
            raise ValueError(f"QLF parameter file {path} has no columns {missing}, its columns are {param_dat.dtype.names}")
        return params

    @staticmethod
    def find_nearest1(array, value):
        """
        Returns the index of the element of an array closest to a value (the first, for ties).


        :param  array:  values to search
        :type   array:  array-like
        :param  value:  value to find the closest element to
        :type   value:  float
        """
        return int(np.argmin(np.abs(np.asarray(array, dtype=float) - value)))

    def get_nearest_LF_z_model(self, z_midplane):
        """
        redshift of the QLF model closest to the midplane redshift of a lightcone redshift shell.

        Returns the QLF redshift

        :param  z_midplane: midplane redshift of the shell
        :type   z_midplane: float
        """
        return float(self.qlf_params['z'][self.find_nearest1(self.qlf_params['z'], z_midplane)])

    def update_qlf_redshift(self, z_midplane):
        """
        Select the parameters of the QLF model at the redshift closest to the shell's midplane redshift.

        :param  z_midplane: midplane redshift of the shell
        :type   z_midplane: float
        """
        self.z_ind = self.find_nearest1(self.qlf_params['z'], z_midplane)
        self.z_qlf = float(self.qlf_params['z'][self.z_ind])
        self.qlf_model = {k: float(v[self.z_ind]) for k, v in self.qlf_params.items()}

    def qlf_values(self, bolometric_luminosities):
        """
        Shen+2020 double power law QLF

        Returns the QLF [Mpc^-3 dex^-1] as an np.ndarray.

        :param  bolometric_luminosities:    bolometric luminosities, erg/s if they have no units
        :type   bolometric_luminosities:    unyt.unyt_array or np.ndarray
        """
        if self.qlf_model is None:
            raise ValueError("No QLF selected. Define the QLF's redshift with update_qlf_redshift.")
        bolometric_luminosities = apply_expected_units(bolometric_luminosities, unyt.erg*unyt.s**-1)
        p = self.qlf_model
        x = 10**(np.log10(bolometric_luminosities.to_value(unyt.Lsun)) - p['log_L_star'])
        return 10**p['log_phi_star'] / (x**p['gamma1'] + x**p['gamma2'])

    @staticmethod
    def _expected_in_bin(qlf_grid_values, log_Lbol_grid_values, log_Lmin, log_Lmax):
        """
        Quasar number density in a luminosity bin, i.e. the QLF integrated (Simpson's rule) over the grid points in the bin.

        Returns the number density [Mpc^-3] as a float.

        :param  qlf_grid_values:        QLF on the grid [Mpc^-3 dex^-1]
        :type   qlf_grid_values:        np.ndarray
        :param  log_Lbol_grid_values:   increasing log10 bolometric luminosities of the grid
        :type   log_Lbol_grid_values:   np.ndarray
        :param  log_Lmin:               lower edge of the bin, log10 bolometric luminosity
        :type   log_Lmin:               float
        :param  log_Lmax:               upper edge of the bin, log10 bolometric luminosity
        :type   log_Lmax:               float
        """
        left = np.searchsorted(log_Lbol_grid_values, log_Lmin, side="left")
        right = np.searchsorted(log_Lbol_grid_values, log_Lmax, side="right")
        return integrate.simpson(y=qlf_grid_values[left:right], x=log_Lbol_grid_values[left:right])

    @staticmethod
    def _cumulative_from_bright_end(qlf_grid_values, log_Lbol_grid_values):
        """
        Number density of quasars brighter than each grid point, i.e. the QLF integrated (trapezoid rule) from the bright
        end of the grid down. 
        Accumulating from the bright end keeps the precision of the small numbers there.

        Returns the cumulative number density [Mpc^-3] at each grid point, decreasing with luminosity.

        :param  qlf_grid_values:        QLF on the grid [Mpc^-3 dex^-1]
        :type   qlf_grid_values:        np.ndarray
        :param  log_Lbol_grid_values:   increasing log10 bolometric luminosities of the grid
        :type   log_Lbol_grid_values:   np.ndarray
        """
        y = np.asarray(qlf_grid_values, dtype=float)
        steps = 0.5 * (y[1:] + y[:-1]) * np.diff(log_Lbol_grid_values)
        return np.append(np.cumsum(steps[::-1])[::-1], 0.)

    @staticmethod
    def _cumulative_at(cumulative, log_Lbol_grid_values, log_L):
        """
        Cumulative number density brighter than log_L, linearly interpolated between grid points.

        Returns the cumulative number density [Mpc^-3], as a float or np.ndarray.

        :param  cumulative:             cumulative number density at the grid points, from _cumulative_from_bright_end
        :type   cumulative:             np.ndarray
        :param  log_Lbol_grid_values:   increasing log10 bolometric luminosities of the grid
        :type   log_Lbol_grid_values:   np.ndarray
        :param  log_L:                  log10 bolometric luminosity
        :type   log_L:                  float or np.ndarray
        """
        return np.interp(log_L, log_Lbol_grid_values, cumulative)

    @staticmethod
    def itr_sample_qlf(qlf_grid_values, Log_Lbol_grid_values, cosmo_volume, sample_size, delta_log_Lbol=0.25,
                       log_Lmin=30., log_Lmax=50., min_expected_numb=1., progress_bar=False, rng=None,
                       within_bin="qlf", exact_bins=True, bright_end="all"):
        """
        Sample log10 bolometric luminosities from a QLF, from the bright end down. 
        The number of quasars in each bin is drawn from a Poisson distribution and their luminosities across the bin, until
        sample_size + 1 luminosities are drawn or log_Lmin is reached. 

        Returns the sampled log10 bolometric luminosities [erg/s].

        :param  qlf_grid_values:        QLF on the grid [Mpc^-3 dex^-1]
        :type   qlf_grid_values:        np.ndarray
        :param  Log_Lbol_grid_values:   increasing log10 bolometric luminosities of the grid
        :type   Log_Lbol_grid_values:   np.ndarray
        :param  cosmo_volume:           comoving volume [Mpc^3]
        :type   cosmo_volume:           float or unyt.unyt_quantity
        :param  sample_size:            number of luminosities to sample
        :type   sample_size:            int
        :param  delta_log_Lbol:         width of the bins in log10 luminosity
        :type   delta_log_Lbol:         float
        :param  log_Lmin:               lowest log10 luminosity to sample down to
        :type   log_Lmin:               float
        :param  log_Lmax:               upper edge of the first bin, log10 bolometric luminosity
        :type   log_Lmax:               float
        :param  min_expected_numb:      expected number of quasars for sampling to start, used if bright_end is
                                        'cumulative' or 'per_bin'
        :type   min_expected_numb:      float
        :param  progress_bar:           If True, show a progress bar of the number of luminosities sampled
        :type   progress_bar:           boolean
        :param  rng:                    random number generator with poisson and random_sample (and uniform) methods,
                                        e.g. np.random.RandomState. If None, numpy's global random state
        :type   rng:                    np.random.RandomState or None
        :param  within_bin:             'qlf': luminosities follow the QLF across the bin (inverse of its cumulative
                                        distribution). 'uniform': uniform in log10 L across the bin. Both use the same
                                        random numbers, so the same quasars only move within their bins
        :type   within_bin:             str
        :param  exact_bins:             If True, bin edges are log_Lmax - k delta_log_Lbol and the expected numbers are
                                        differences of the cumulative QLF, so the bins add up to the total exactly. If
                                        False, the original method (edges stepped by repeated subtraction, Simpson's rule
                                        over the grid points in each bin)
        :type   exact_bins:             boolean
        :param  bright_end:             'all': every bin from log_Lmax down is sampled, so rare bright quasars are
                                        drawn with their Poisson probability, independent of the bin width
                                        (min_expected_numb is not used). 'cumulative': sampling starts at the
                                        luminosity where the expected number of brighter quasars in the volume, N(>L) V,
                                        reaches min_expected_numb, independent of the bin width. 'per_bin': bins are
                                        skipped until one bin alone expects min_expected_numb (the original method)
        :type   bright_end:             str
        """
        if bright_end not in ("all", "cumulative", "per_bin"):
            raise ValueError(f"unknown bright_end '{bright_end}', use 'all', 'cumulative' or 'per_bin'")
        if within_bin not in ("qlf", "uniform"):
            raise ValueError(f"unknown within_bin '{within_bin}', use 'qlf' or 'uniform'")
        rng = np.random if rng is None else rng
        sample_size = int(sample_size)
        volume = apply_expected_units(cosmo_volume, unyt.Mpc**3).to_value(unyt.Mpc**3)
        target_val = min_expected_numb / volume

        cumulative = None
        if exact_bins or within_bin == "qlf" or bright_end == "cumulative":
            cumulative = SampleQLF._cumulative_from_bright_end(qlf_grid_values, Log_Lbol_grid_values)
            # inverse of the cumulative distribution: luminosity as a function of the number density brighter than it
            cumulative_increasing, grid_reversed = cumulative[::-1], Log_Lbol_grid_values[::-1]
        cumulative_at = lambda log_L: SampleQLF._cumulative_at(cumulative, Log_Lbol_grid_values, log_L)

        if bright_end == "cumulative":
            # start where the expected number of brighter AGN in the volume reaches min_expected_numb
            log_Lmax = min(log_Lmax, float(np.interp(target_val, cumulative_increasing, grid_reversed)))

        sampled_log_Lbol = np.empty(sample_size+1, dtype=float)
        n_filled = 0         # values stored
        running_total = 0    # values drawn
        sampling = bright_end != "per_bin"  # for 'per_bin', set once a bin reaches the expected number
        k = 0                # bin number, counted from the bright end
        bin_max = log_Lmax
        bin_min = log_Lmax - delta_log_Lbol

        pbar = tqdm(total=sample_size+1, disable=not progress_bar)
        while running_total < sample_size and bin_min > log_Lmin:
            if exact_bins:
                n_brighter_max, n_brighter_min = cumulative_at(bin_max), cumulative_at(bin_min)
                expected_in_bin = n_brighter_min - n_brighter_max
            else:
                expected_in_bin = SampleQLF._expected_in_bin(qlf_grid_values, Log_Lbol_grid_values, bin_min, bin_max)
            if not sampling and expected_in_bin >= target_val:
                sampling = True
            if sampling:
                n_poisson = rng.poisson(lam=expected_in_bin * volume)
                if n_poisson > 0:
                    if within_bin == "uniform":
                        drawn = rng.uniform(low=bin_min, high=bin_max, size=n_poisson)
                    else:
                        # same random numbers as uniform(bin_min, bin_max), mapped through the bin's cumulative
                        # distribution: u=0 at the faint edge, u=1 at the bright edge
                        u = rng.random_sample(n_poisson)
                        n_at_min, n_at_max = cumulative_at(bin_min), cumulative_at(bin_max)
                        drawn = np.interp(n_at_min - u * (n_at_min - n_at_max), cumulative_increasing, grid_reversed)
                    n_keep = min(n_poisson, sample_size + 1 - n_filled)
                    sampled_log_Lbol[n_filled:n_filled+n_keep] = drawn[:n_keep]
                    n_filled += n_keep
                    running_total += n_poisson
                    pbar.update(n_keep)
            # next bin down
            k += 1
            if exact_bins:
                bin_max = log_Lmax - k * delta_log_Lbol
                bin_min = log_Lmax - (k + 1) * delta_log_Lbol
            else:
                bin_min -= delta_log_Lbol
                bin_max -= delta_log_Lbol
        pbar.close()
        return sampled_log_Lbol[:n_filled]

    def shell_rng(self, z_midplane):
        """
        Random number generator of a shell, seeded from the seed and the exact value of the shell's midplane redshift,
        so a shell's draws do not depend on which shells were sampled before.

        Returns an np.random.RandomState.

        :param  z_midplane: midplane redshift of the shell
        :type   z_midplane: float
        """
        z_bits = np.array([z_midplane], dtype=np.float64).view(np.uint32)
        return np.random.RandomState(np.concatenate([[self.seed % 2**32], z_bits]).astype(np.uint32))

    def sample_LF_Lbol(self, z_midplane, n_BHs, cosmo_volume, log_Lbol_bin_size=0.1, log_Lbol_min_value=30, log_Lbol_max_value=50, min_expected=1., progress_bar=False,
                       per_shell_seed=True, within_bin="qlf", exact_bins=True, bright_end="all"):
        """
        Sample bolometric luminosities from the QLF, for the comoving volume of the shell.

        Returns a tuple 
            (
                sampled bolometric luminosities [erg/s] in descending order, 
                number of values sampled
            ).
        
        Normally n_BHs + 1 values, fewer if the luminosity range runs out first. 

        :param  z_midplane:         midplane redshift of the shell, selecting its QLF (see update_qlf_redshift)
        :type   z_midplane:         float
        :param  n_BHs:              number of bolometric luminosities to sample
        :type   n_BHs:              int
        :param  cosmo_volume:       comoving volume of the shell [Mpc^3]
        :type   cosmo_volume:       float or unyt.unyt_quantity
        :param  log_Lbol_bin_size:  width of the sampling bins in log10 luminosity
        :type   log_Lbol_bin_size:  float
        :param  log_Lbol_min_value: minimum log10 bolometric luminosity [erg/s] to sample
        :type   log_Lbol_min_value: float
        :param  log_Lbol_max_value: maximum log10 bolometric luminosity [erg/s] that can be sampled
        :type   log_Lbol_max_value: float
        :param  min_expected:       expected number of quasars for sampling to start if bright_end is 'cumulative' or
                                    'per_bin', see itr_sample_qlf
        :type   min_expected:       float
        :param  progress_bar:       If True, show a progress bar
        :type   progress_bar:       boolean
        :param  per_shell_seed:     If True, use a random number generator seeded from the seed and z_midplane (shell_rng), so the
                                    shell's luminosities do not depend on other calls. If False, continue the instance's random
                                    stream (the original method)
        :type   per_shell_seed:     boolean
        :param  within_bin:         how luminosities are spread within a bin, 'qlf' or 'uniform', see itr_sample_qlf
        :type   within_bin:         str
        :param  exact_bins:         If True, exact bin edges and expected numbers from the cumulative QLF, see itr_sample_qlf
        :type   exact_bins:         boolean
        :param  bright_end:         where sampling starts at the bright end, 'all', 'cumulative' or 'per_bin', see
                                    itr_sample_qlf
        :type   bright_end:         str
        """
        self.update_qlf_redshift(z_midplane)

        # fine grid of the QLF to integrate over, spacing ~1/1000 of the bin size
        n_qlf_grid = int((log_Lbol_max_value-log_Lbol_min_value)//(log_Lbol_bin_size/1000)) + 1
        log_Lbol_grid = np.linspace(log_Lbol_min_value-(0.5*log_Lbol_bin_size), log_Lbol_max_value+(0.5*log_Lbol_bin_size), n_qlf_grid)
        qlf_grid = self.qlf_values(unyt.unyt_array(10**log_Lbol_grid, units=unyt.erg*unyt.s**-1))

        sampled_log_Lbol = self.itr_sample_qlf(
            qlf_grid_values=qlf_grid,
            Log_Lbol_grid_values=log_Lbol_grid,
            cosmo_volume=cosmo_volume, sample_size=n_BHs,
            delta_log_Lbol=log_Lbol_bin_size,
            log_Lmax=log_Lbol_grid[-1],
            log_Lmin=log_Lbol_grid[0],
            min_expected_numb=min_expected,
            progress_bar=progress_bar,
            rng=self.shell_rng(z_midplane) if per_shell_seed else self.rng,
            within_bin=within_bin,
            exact_bins=exact_bins,
            bright_end=bright_end,
        )

        # contiguous copy: numpy's power can differ in the last bit for a reversed view
        descending = np.ascontiguousarray(np.sort(sampled_log_Lbol)[::-1])
        ordered_sample = unyt.unyt_array(10**descending, units=unyt.erg * unyt.s**-1)
        return ordered_sample, len(ordered_sample)

