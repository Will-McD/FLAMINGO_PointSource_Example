#!/bin/env python
"""
X-ray spectra of AGN point sources with PyXspec, and the tables used to give lightcone AGN their X-ray fluxes: the
Ueda+2014 column density distribution (sampled for each AGN) and the broadband flux of the AGN spectrum as a function
of redshift and column density, normalised to the flux of its intrinsic power law.

    AGNSpectra          spectra of an AGN model through a telescope's response, see Spectrum
    PointSourceTables   building, writing and reading the column density and broadband flux tables
    FluxTable           a broadband flux table, evaluated for any number of particles
"""
import os
import datetime
import h5py
import numpy as np
import unyt
from unyt.exceptions import UnytError
from scipy import integrate
from scipy.interpolate import PchipInterpolator
from numba import njit, prange
from tqdm import tqdm
import xspec

from swiftlet.snapshot_units import apply_expected_units
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone

from .arf_utils import read_ARF
from .sample_qlf import SHEN2020_COEFFS
from .telescope_utils import PRESET_TELESCOPES, FLAMINGO_BANDS

xspec.Xset.chatter=0 # make output from xspec less verbose
xspec.Xset.logChatter=0 # turn off logged output from xspec

# directory tables are written to when no other directory is given
TABLE_DIR = os.path.join(os.path.dirname(__file__), "AGN_tables")


# the xspec model last made by AGNSpectra.compute_spectra with reuse_model=True. xspec has one active (unnamed)
# model, so it is kept here and updated with new parameter values instead of being made again
_XSPEC_MODEL_CACHE = {"key": None, "model": None, "params": {}}

# units of the spectra before the effective area is applied, number of photons/s/cm^2/keV. unyt treats
# photons as dimensionless and drops them in products, so they are left out to keep the units consistent.
SPECTRUM_UNITS = unyt.s**-1 * unyt.cm**-2 * unyt.keV**-1

# lowest bits of the float64 mantissas of Lbol and z dropped when they are used to identify particles in sample_nH,
# 20 of 52 bits leaves a relative precision of ~1e-10
KEY_DROPPED_BITS = 20


def configure_xspec(snapshot_filename, abund="aspl", xsect="vern", xspec_obj=None):
    """
    Set the xspec cosmology from a snapshot, and the abundance table and photoionisation cross sections.
    Call this before computing spectra, otherwise the xspec defaults are used (e.g. 'angr' abundances).

    Returns the snapshot cosmology object, a lightcone_io.xray_utils.Snapshot_Cosmology_For_Lightcone.

    :param  snapshot_filename:  path to a FLAMINGO snapshot .hdf5 file, or to a snapshot directory
    :type   snapshot_filename:  str
    :param  abund:              xspec abundance table, e.g. 'aspl' (Asplund+09) or 'wilm' (Wilms+00)
    :type   abund:              str
    :param  xsect:              xspec photoionisation cross sections, e.g. 'vern' (Verner+96)
    :type   xsect:              str
    :param  xspec_obj:          PyXspec module to configure. If None, the imported xspec module
    :type   xspec_obj:          module or None
    """
    xspec_obj = xspec if xspec_obj is None else xspec_obj
    cosmo = Snapshot_Cosmology_For_Lightcone(snapshot_filename)
    H0_sim=cosmo.raw_cosmo['H0 [internal units]'][0] # in km/Mpc/s
    Omega_Lambda=cosmo.raw_cosmo['Omega_lambda'][0] # dimensionless

    xspec_obj.Xset.allowPrompting = False # keeps pyxspec from hanging, waiting for a response to a prompt
    xspec_obj.Xset.allowNewAttributes=True # allow new attributes
    xspec_obj.Xset.abund = abund
    xspec_obj.Xset.xsect = xsect
    xspec_obj.Xset.cosmo=f"{H0_sim:.2f},{0.},{Omega_Lambda:f}" # "H0, q0, l0" # km/s/Mpc, decelleration parameter

    xspec_obj.Xset.show()
    xspec_obj.Xset.chatter=0 # make output from xspec less verbose
    xspec_obj.Xset.logChatter=0 # turn off logged output from xspec
    return cosmo


def is_power_of_two(n):
    """
    Check if an integer is a power of two.

    Returns True if n is a power of two.

    :param  n:  integer to check
    :type   n:  int
    """
    return n > 0 and (n & (n-1)) == 0


def _centred_bins(vmin, vmax, width):
    """
    Regular bins centred on vmin, vmin + width, ..., vmax. Values are rounded to 10 decimal places so
    that bin centres meant to fall on a boundary (e.g. log10 nH = 21) are not moved across it by rounding errors.

    Returns a tuple of (bin edges, bin centres), each an np.ndarray.

    :param  vmin:   centre of the first bin
    :type   vmin:   float
    :param  vmax:   centre of the last bin
    :type   vmax:   float
    :param  width:  width of the bins
    :type   width:  float
    """
    n = int(round((vmax-vmin)/width)) + 1
    mids = np.round(np.linspace(vmin, vmax, n), 10)
    edges = np.round(np.linspace(vmin-width/2, vmax+width/2, n+1), 10)
    return edges, mids


def _regular_edges(vmin, vmax, width):
    """
    Edges of regular bins from vmin to vmax.

    Returns the bin edges as an np.ndarray.

    :param  vmin:   lower edge of the first bin
    :type   vmin:   float
    :param  vmax:   upper edge of the last bin
    :type   vmax:   float
    :param  width:  width of the bins
    :type   width:  float
    """
    return np.round(np.linspace(vmin, vmax, int(round((vmax-vmin)/width)) + 1), 10)


def _powerlaw_params(redshift, spec_index, norm_fact):
    """
    Parameters of a zpowerlw component.

    Returns the parameters as a dictionary.

    :param  redshift:       redshift of the AGN point source
    :type   redshift:       float
    :param  spec_index:     photon index of the power law
    :type   spec_index:     float
    :param  norm_fact:      normalisation of the power law
    :type   norm_fact:      float
    """
    return {"PhoIndex":spec_index, "Redshift":redshift, "norm":norm_fact}


def _pexmon_params(redshift, spec_index, norm_fact, rel_refl, cos_inc, fold_energy, ratio_abund, Fe_abund):
    """
    Parameters of a pexmon component, rel_refl is frozen.

    Returns the parameters as a dictionary.

    :param  redshift:       redshift of the AGN point source
    :type   redshift:       float
    :param  spec_index:     photon index of the power law
    :type   spec_index:     float
    :param  norm_fact:      normalisation of the power law
    :type   norm_fact:      float
    :param  rel_refl:       relative strength of the reflection, -1 gives only the reflected component
    :type   rel_refl:       float
    :param  cos_inc:        cosine of the AGN inclination angle
    :type   cos_inc:        float
    :param  fold_energy:    cut-off energy of the power law [keV]
    :type   fold_energy:    float
    :param  ratio_abund:    abundance of elements heavier than He relative to solar
    :type   ratio_abund:    float
    :param  Fe_abund:       iron abundance relative to solar
    :type   Fe_abund:       float
    """
    return {
        "PhoIndex":spec_index, "redshift":redshift, "norm":norm_fact,
        "rel_refl":f"{rel_refl},,-2,-2,,", "Incl":cos_inc, "foldE":fold_energy,
        "abund":ratio_abund, "Fe_abund":Fe_abund,
    }


def _integrate_band(spectrum_x, spectrum_y, min_energy, max_energy, bin_method, energy_flux):
    """
    Integrate a spectrum between two photon energies. Romberg integration is used for 2^k+1 evenly
    spaced photon energies, Simpson's rule otherwise.

    Returns the energy flux [erg/s/cm^2] if energy_flux is True, else the photon flux [photons/s/cm^2,
    given as 1/s/cm^2], as a unyt.unyt_quantity (times the effective area units, if included in the spectrum).

    :param  spectrum_x:     observed photon energies [keV]
    :type   spectrum_x:     unyt.unyt_array
    :param  spectrum_y:     spectrum at each photon energy [photons/s/cm^2/keV]
    :type   spectrum_y:     unyt.unyt_array
    :param  min_energy:     lower limit of the integral [keV]
    :type   min_energy:     unyt.unyt_quantity
    :param  max_energy:     upper limit of the integral [keV]
    :type   max_energy:     unyt.unyt_quantity
    :param  bin_method:     spacing of the photon energies, 'linear' or 'log'
    :type   bin_method:     str
    :param  energy_flux:    If True, integrate E * spectrum for the energy flux
    :type   energy_flux:    boolean
    """
    if spectrum_x.max()<=min_energy or spectrum_x.min()>=max_energy:
        raise ValueError("Spectrum is not within range of values given for numerical intergration")
    # tolerance, so photon energies on the limits are not lost to rounding errors
    tol = 1e-9 * (max_energy - min_energy)
    mE = (spectrum_x>=min_energy-tol) & (spectrum_x<=max_energy+tol)
    energy_in_band = spectrum_x[mE].to_value("keV")
    integrand = spectrum_y[mE]
    if energy_flux:
        integrand = integrand * spectrum_x[mE]

    dE = np.diff(energy_in_band)
    if bin_method=="linear" and is_power_of_two(len(energy_in_band)-1) and np.allclose(dE, dE[0]):
        value = integrate.romb(y=integrand.value, dx=dE[0])
    else:
        value = integrate.simpson(y=integrand.value, x=energy_in_band)

    flux = unyt.unyt_quantity(value, units=integrand.units * unyt.keV)
    if energy_flux:
        flux = flux.to(flux.units * unyt.erg / unyt.keV)
    return flux


class Spectrum:
    """
    A computed AGN spectrum: the photon energies, the spectrum before the effective area (intrinsic_values) and
    the effective area, with the band, model and parameters it was computed for. Operations return new
    Spectrum objects, a Spectrum is never changed.
    """
    def __init__(self, photon_energies, intrinsic_values, effective_area, bin_method, energy_band,
                 normalisation_factor=1., model=None, parameters=None, telescope_name=None):
        """
        Store a spectrum.

        :param  photon_energies:        observed photon energies [keV]
        :type   photon_energies:        unyt.unyt_array
        :param  intrinsic_values:       spectrum before the effective area [photons/s/cm^2/keV]
        :type   intrinsic_values:       unyt.unyt_array
        :param  effective_area:         effective area at each photon energy (dimensionless ones for no telescope)
        :type   effective_area:         unyt.unyt_array
        :param  bin_method:             spacing of the photon energies, 'linear' or 'log'
        :type   bin_method:             str
        :param  energy_band:            (minimum, maximum) observed photon energy of the band [keV]
        :type   energy_band:            tuple of unyt.unyt_quantity
        :param  normalisation_factor:   factor the spectrum has been scaled by
        :type   normalisation_factor:   float or unyt.unyt_quantity (dimensionless)
        :param  model:                  spectrum model the spectrum was computed from
        :type   model:                  str, xspec.Model, callable or None
        :param  parameters:             parameters of the model
        :type   parameters:             any
        :param  telescope_name:         name of the telescope of the effective area
        :type   telescope_name:         str or None
        """
        self.photon_energies = photon_energies
        self.intrinsic_values = intrinsic_values.to(SPECTRUM_UNITS)
        self.effective_area = effective_area
        self.bin_method = bin_method
        self.energy_band = energy_band
        self.normalisation_factor = apply_expected_units(normalisation_factor, unyt.dimensionless)
        self.model = model
        self.parameters = parameters
        self.telescope_name = telescope_name

    @property
    def values(self):
        """spectrum including the effective area [photons/s/cm^2/keV, times the effective area units]"""
        units = SPECTRUM_UNITS if self.effective_area.units==unyt.dimensionless else SPECTRUM_UNITS * self.effective_area.units
        return (self.intrinsic_values * self.effective_area).to(units)

    def _bounds(self, min_energy, max_energy):
        """
        Photon energy range in keV, the band limits are used where a value is None.

        Returns a tuple of (minimum, maximum) photon energy, each a unyt.unyt_quantity.

        :param  min_energy:     minimum photon energy [keV]
        :type   min_energy:     float, unyt.unyt_quantity or None
        :param  max_energy:     maximum photon energy [keV]
        :type   max_energy:     float, unyt.unyt_quantity or None
        """
        min_energy = self.energy_band[0] if min_energy is None else apply_expected_units(min_energy, unyt.keV)
        max_energy = self.energy_band[1] if max_energy is None else apply_expected_units(max_energy, unyt.keV)
        return min_energy, max_energy

    def flux(self, min_energy=None, max_energy=None, effective_area=True):
        """
        Energy and photon flux of the spectrum.

        Returns a tuple of (energy flux [erg/s/cm^2], photon flux [photons/s/cm^2, given as 1/s/cm^2]), each a
        unyt.unyt_quantity, times the effective area units if effective_area is True.

        :param  min_energy:         lower limit of the integral [keV]. If None, the band minimum
        :type   min_energy:         float, unyt.unyt_quantity or None
        :param  max_energy:         upper limit of the integral [keV]. If None, the band maximum
        :type   max_energy:         float, unyt.unyt_quantity or None
        :param  effective_area:     If True, integrate the spectrum including the effective area, else intrinsic_values
        :type   effective_area:     boolean
        """
        min_energy, max_energy = self._bounds(min_energy, max_energy)
        spectrum = self.values if effective_area else self.intrinsic_values
        return (_integrate_band(self.photon_energies, spectrum, min_energy, max_energy, self.bin_method, True),
                _integrate_band(self.photon_energies, spectrum, min_energy, max_energy, self.bin_method, False))

    def scaled(self, factor):
        """
        The spectrum multiplied by a factor.

        Returns a new Spectrum, its normalisation_factor includes the factor.

        :param  factor:     factor to scale the spectrum by
        :type   factor:     float or unyt.unyt_quantity (dimensionless)
        """
        factor = apply_expected_units(factor, unyt.dimensionless)
        return Spectrum(self.photon_energies, self.intrinsic_values * factor, self.effective_area, self.bin_method,
                        self.energy_band, self.normalisation_factor * factor, self.model, self.parameters, self.telescope_name)

    def normalised_to(self, target_flux, min_energy=None, max_energy=None):
        """
        The spectrum scaled so that its flux between two photon energies is target_flux.

        Returns a new Spectrum.

        :param  target_flux:    (value, units) of the target flux. The units give a photon ('photon/s/cm**2') or
                                energy ('erg/s/cm**2') flux before the effective area, or after it ('photon/s', 'erg/s')
        :type   target_flux:    tuple
        :param  min_energy:     lower limit of the flux [keV]. If None, the band minimum
        :type   min_energy:     float, unyt.unyt_quantity or None
        :param  max_energy:     upper limit of the flux [keV]. If None, the band maximum
        :type   max_energy:     float, unyt.unyt_quantity or None
        """
        target = target_flux[0] * unyt.Unit(target_flux[1])
        min_energy, max_energy = self._bounds(min_energy, max_energy)
        # photon / energy flux, before / after the effective area
        for effective_area in (False, True):
            spectrum = self.values if effective_area else self.intrinsic_values
            for energy_flux in (False, True):
                flux_units = spectrum.units * unyt.keV * (unyt.keV if energy_flux else unyt.dimensionless)
                if target.units.dimensions == flux_units.dimensions:
                    flux = _integrate_band(self.photon_energies, spectrum, min_energy, max_energy, self.bin_method, energy_flux)
                    return self.scaled((target / flux).to(unyt.dimensionless))
        raise UnytError(f"target flux units ({target.units}) do not match a photon or energy flux of the spectrum")


class AGNSpectra:
    """
    Spectra of AGN point sources, from xspec models or functions, with the effective area of a telescope
    and integrated over the observer's energy band.
    """
    def __init__(self,
                    telescope="none",
                    ARF_filename=None,
                    ARF_reader=None,
                    E_band_min=0.5*unyt.keV,
                    E_band_max=2.0*unyt.keV):
        """
        Set up the telescope response and the energy band.

        :param  telescope:      name of the telescope. 'none' applies no effective area. 'ROSAT' and 'eROSITA'
                                read their preset response functions with read_ARF. Any other name needs
                                ARF_filename and/or ARF_reader
        :type   telescope:      str
        :param  ARF_filename:   None for a preset telescope. 'analytic' if ARF_reader is an analytic function
                                of the effective area [cm^2] of the observed photon energy [keV], otherwise
                                the name of the ARF file, read with read_ARF
        :type   ARF_filename:   str or None
        :param  ARF_reader:     None for a preset telescope. Function used by read_ARF to read ARF_filename, or
                                the effective area [cm^2] as a function of photon energy [keV] if
                                ARF_filename='analytic'
        :type   ARF_reader:     callable or None
        :param  E_band_min:     minimum observed photon energy of the band [keV]
        :type   E_band_min:     float or unyt.unyt_quantity
        :param  E_band_max:     maximum observed photon energy of the band [keV]
        :type   E_band_max:     float or unyt.unyt_quantity
        """
        self.energy_band_limits=(apply_expected_units(E_band_min, unyt.keV), apply_expected_units(E_band_max, unyt.keV))
        self._set_telescope(telescope, ARF_filename, ARF_reader)

        # the last spectrum computed, also returned by compute_spectra
        self.last_spectrum=None

    def _set_telescope(self, telescope, ARF_filename=None, ARF_reader=None):
        """
        Define the effective area [cm^2] as a function of observed photon energy [keV].

        :param  telescope:      name of the telescope, see __init__
        :type   telescope:      str
        :param  ARF_filename:   name of the ARF file, 'analytic' or None, see __init__
        :type   ARF_filename:   str or None
        :param  ARF_reader:     ARF reading function or analytic effective area, see __init__
        :type   ARF_reader:     callable or None
        """
        if telescope=="none":
            self._Aeff_units=unyt.dimensionless
            self._response_function=lambda E: np.ones_like(np.asarray(E, dtype=float))
        elif ARF_filename=="analytic":
            if ARF_reader is None:
                raise ValueError("ARF_filename='analytic' needs the effective area function as ARF_reader")
            self._Aeff_units=unyt.cm**2
            self._response_function=ARF_reader
        elif (telescope in PRESET_TELESCOPES) or (ARF_filename is not None) or (ARF_reader is not None):
            self._Aeff_units=unyt.cm**2
            self._response_function=self._ARF_response(
                telescope if ARF_filename is None else ARF_filename, ARF_reader,
                self.energy_band[0].to_value("keV"), self.energy_band[1].to_value("keV"))
        else:
            raise ValueError(f"unknown telescope '{telescope}', give ARF_filename and/or ARF_reader for a new telescope")

        # stored for messages, saving data and rebuilding the response when the band changes
        self._telescope_name=telescope
        self._ARF_filename=ARF_filename
        self._ARF_reader=ARF_reader

    @staticmethod
    def _ARF_response(ARF_filename, ARF_reader, E_min, E_max):
        """
        PCHIP interpolation of the effective area read from an ARF file, zero outside of the band.
        The ARF points either side of the band are kept so that the band edges are interpolated.

        Returns the effective area [cm^2] as a function of photon energy [keV].

        :param  ARF_filename:   name of the ARF file
        :type   ARF_filename:   str
        :param  ARF_reader:     function used by read_ARF to read the file
        :type   ARF_reader:     callable or None
        :param  E_min:          minimum photon energy of the band [keV]
        :type   E_min:          float
        :param  E_max:          maximum photon energy of the band [keV]
        :type   E_max:          float
        """
        energy_centres, effective_areas = read_ARF(ARF_filename, ARF_reader)
        energy_centres = np.asarray(energy_centres, dtype=float)
        effective_areas = np.asarray(effective_areas, dtype=float)

        inside = np.nonzero((energy_centres >= E_min) & (energy_centres <= E_max))[0]
        if len(inside)==0:
            raise ValueError(f"no ARF energies in the band ({E_min}, {E_max}) keV")
        keep = slice(max(inside[0]-1, 0), min(inside[-1]+2, len(energy_centres)))
        f_Aeff = PchipInterpolator(x=energy_centres[keep], y=effective_areas[keep], extrapolate=False)

        def response(E):
            """
            Returns the effective area [cm^2] at each photon energy, zero outside of the band.

            :param  E:  photon energies [keV]
            :type   E:  float or np.ndarray
            """
            E = np.asarray(E, dtype=float)
            return np.where((E >= E_min) & (E <= E_max), np.nan_to_num(f_Aeff(E), nan=0.), 0.)
        return response

    def update_telescope_properties(self, telescope, ARF_filename=None, ARF_reader=None):
        """
        Change the telescope given when initialising the class.

        :param  telescope:      name of the telescope, see __init__
        :type   telescope:      str
        :param  ARF_filename:   name of the ARF file, 'analytic' or None, see __init__
        :type   ARF_filename:   str or None
        :param  ARF_reader:     ARF reading function or analytic effective area, see __init__
        :type   ARF_reader:     callable or None
        """
        self._set_telescope(telescope, ARF_filename, ARF_reader)

    def update_energy_band(self, E_band_min, E_band_max):
        """
        Change the energy band, the response function is rebuilt for the new band.

        :param  E_band_min:     minimum observed photon energy of the band [keV]
        :type   E_band_min:     float or unyt.unyt_quantity
        :param  E_band_max:     maximum observed photon energy of the band [keV]
        :type   E_band_max:     float or unyt.unyt_quantity
        """
        self.energy_band_limits=(apply_expected_units(E_band_min, unyt.keV), apply_expected_units(E_band_max, unyt.keV))
        self._set_telescope(self._telescope_name, self._ARF_filename, self._ARF_reader)

    @property
    def last_model(self):
        """the last spectrum model and its parameters"""
        if self.last_spectrum is None:
            return None, None
        return self.last_spectrum.model, self.last_spectrum.parameters

    @property
    def photon_energies(self):
        """photon energies of the last spectrum [keV], None before a spectrum is computed"""
        return None if self.last_spectrum is None else self.last_spectrum.photon_energies

    @property
    def spectrum_values(self):
        """last spectrum including the effective area, None before a spectrum is computed"""
        return None if self.last_spectrum is None else self.last_spectrum.values

    @property
    def bin_method(self):
        """spacing of the photon energies of the last spectrum, None before a spectrum is computed"""
        return None if self.last_spectrum is None else self.last_spectrum.bin_method

    @property
    def energy_band(self):
        """minimum and maximum observed photon energies of the band [keV]"""
        return (apply_expected_units(self.energy_band_limits[0], unyt.keV), apply_expected_units(self.energy_band_limits[1], unyt.keV))

    @property
    def telescope_name(self):
        """name of the telescope"""
        return self._telescope_name

    @property
    def response_function(self):
        """effective area as a function of observed photon energy [keV]"""
        return self._response_function

    @property
    def broadband_flux(self):
        """(energy flux, photon flux) of the last spectrum in the band, (None, None) before a spectrum is computed"""
        if self.last_spectrum is None:
            return (None, None)
        return self.last_spectrum.flux()

    @property
    def last_normalisation_factor(self):
        """factor the last spectrum was scaled by"""
        if self.last_spectrum is None:
            return unyt.unyt_quantity(1., units=unyt.dimensionless)
        return self.last_spectrum.normalisation_factor

    @staticmethod
    def analytic_plaw_flux_energy(a, b, z, Gamma, A=1):
        """
        Analytic energy flux of the power law A * (E * (1+z))^-Gamma between a and b.

        Returns the de-absorbed intrinsic energy flux [keV/s/cm^2 for A in photon/keV/s/cm^2] as a float.

        :param  a:      minimum observed photon energy [keV]
        :type   a:      float
        :param  b:      maximum observed photon energy [keV]
        :type   b:      float
        :param  z:      redshift of the AGN point source
        :type   z:      float
        :param  Gamma:  photon index
        :type   Gamma:  float
        :param  A:      normalisation of the power law
        :type   A:      float
        """
        return A * (b**(2-Gamma) - a**(2-Gamma))/((2-Gamma) * (1+z)**Gamma)

    @staticmethod
    def _set_pyxspec_parameters(m, parameters, previous=None):
        """
        Set the parameters of a pyxspec model. If previous is given, only parameters whose values differ from it
        are set, so xspec does not see a change in the others.

        Returns the values of the parameters given, as a dictionary keyed by parameter number or (component, parameter name).

        :param  m:              pyxspec model
        :type   m:              xspec.Model
        :param  parameters:     parameter values, see compute_spectra
        :type   parameters:     dict or None
        :param  previous:       parameter values already set in the model, as returned by a previous call
        :type   previous:       dict or None
        """
        if parameters is None:
            return {}
        if not isinstance(parameters, dict):
            raise TypeError("parameters is not a dictionary")
        previous = {} if previous is None else previous
        changed = lambda key, value: key not in previous or previous[key] != value

        if all(isinstance(k, int) for k in parameters):
            # items are numbered by the model parameters, pyxspec is 1 indexed
            if max(parameters) > m.nParameters:
                raise ValueError(f"Cannot set parameters > number of model parameters, max={m.nParameters}")
            if min(parameters) <= 0:
                raise ValueError("Cannot set parameters <= 0, PyXspec model parameters are 1 indexed")
            to_set = {k: v for k, v in parameters.items() if changed(k, v)}
            if to_set:
                m.setPars(to_set)
            return dict(parameters)

        if not all(isinstance(k, str) for k in parameters):
            raise TypeError("items of parameter dictionary are a mix of integers and strings")

        # items are model components
        unknown = [k for k in parameters if k not in m.componentNames]
        if unknown:
            raise ValueError(f"components {unknown} are not in the model, its components are {m.componentNames}")
        values_set = {}
        for comp_name, values in parameters.items():
            comp = getattr(m, comp_name)
            param_names = comp.parameterNames
            if isinstance(values, dict):
                # match by parameter name, set in the order of the model
                unknown = [k for k in values if k not in param_names]
                if unknown:
                    raise ValueError(f"parameters {unknown} are not in the {comp_name} component, its parameters are {param_names}")
                pairs = [(name, values[name]) for name in param_names if name in values]
            elif isinstance(values, (list, tuple, np.ndarray)):
                # match by index
                if len(values)!=len(param_names):
                    raise ValueError(f"incorrect number of parameters assigned to {comp_name} component, it has {param_names}")
                pairs = list(zip(param_names, values))
            else:
                raise TypeError(f"parameters of {comp_name} must be a dictionary or a sequence")
            for param_name, value in pairs:
                if changed((comp_name, param_name), value):
                    setattr(comp, param_name, value)
                values_set[(comp_name, param_name)] = value
        return values_set

    @staticmethod
    def _xspec_energy_grid(min_energy, max_energy, nbins, bin_method):
        """
        xspec energy grid of nbins bins, padded by half a bin either side so that the bin centres are the same
        photon energies as get_observer_energies, from min_energy to max_energy. A linear grid that would start
        at <= 0 keV is not padded.

        Returns the grid as a string for xspec.AllModels.setEnergies.

        :param  min_energy:     minimum photon energy of the bin centres [keV]
        :type   min_energy:     unyt.unyt_quantity
        :param  max_energy:     maximum photon energy of the bin centres [keV]
        :type   max_energy:     unyt.unyt_quantity
        :param  nbins:          number of bins
        :type   nbins:          int
        :param  bin_method:     spacing of the bins, 'linear' or 'log'
        :type   bin_method:     str
        """
        E0, E1 = min_energy.to_value("keV"), max_energy.to_value("keV")
        if bin_method=="linear":
            half_bin = 0.5*(E1-E0)/(nbins-1)
            lo, hi = (E0-half_bin, E1+half_bin) if E0-half_bin > 0 else (E0, E1)
            return f"{lo!r} {hi!r} {nbins:d} lin"
        half_bin = (E1/E0)**(0.5/(nbins-1))
        return f"{E0/half_bin!r} {E1*half_bin!r} {nbins:d} log"

    @staticmethod
    def _pyxspec_spectrum(m, spec_idx, bin_method="linear"):
        """
        Spectrum of a pyxspec model at the centres of its photon energy bins, the arithmetic centres for
        linear bins and the geometric centres for log bins.

        Returns a tuple of (observed photon energies [keV], spectrum [photons/s/cm^2/keV]), each a unyt.unyt_array.

        :param  m:          pyxspec model
        :type   m:          xspec.Model
        :param  spec_idx:   spectrum number of the model
        :type   spec_idx:   int
        :param  bin_method: spacing of the bins, 'linear' or 'log'
        :type   bin_method: str
        """
        bin_edges = np.asarray(m.energies(spec_idx), dtype=float)
        if bin_method=="log":
            centres = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        else:
            centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
        x = unyt.unyt_array(centres, units=unyt.keV)
        y = unyt.unyt_array(np.asarray(m.values(spec_idx), dtype=float)/np.diff(bin_edges), units=SPECTRUM_UNITS)
        return x, y

    def _energy_bounds(self, min_energy, max_energy):
        """
        Photon energy range in keV, the band limits are used where a value is None.

        Returns a tuple of (minimum, maximum) photon energy, each a unyt.unyt_quantity.

        :param  min_energy:     minimum photon energy [keV]
        :type   min_energy:     float, unyt.unyt_quantity or None
        :param  max_energy:     maximum photon energy [keV]
        :type   max_energy:     float, unyt.unyt_quantity or None
        """
        band_min, band_max = self.energy_band
        min_energy = band_min if min_energy is None else apply_expected_units(min_energy, unyt.keV)
        max_energy = band_max if max_energy is None else apply_expected_units(max_energy, unyt.keV)
        return min_energy, max_energy

    @staticmethod
    def spectrum_template_type(template):
        """
        Type of spectrum template.

        Returns 'xspec_str' for an xspec model string, 'pyxspec_model' for a pyxspec model and
        'callable_function' for a function.

        :param  template:   spectrum model, see compute_spectra
        :type   template:   str, xspec.Model or callable
        """
        if isinstance(template, str):
            return "xspec_str"
        if "xspec" in type(template).__module__:
            return "pyxspec_model"
        if callable(template):
            return "callable_function"
        raise TypeError("can not recognise spectrum model type")

    def get_observer_energies(self, nbins, E0=None, E1=None, bin_method=None):
        """
        Photon energies between two limits, the energy band by default.

        Returns the photon energies [keV] as a unyt.unyt_array.

        :param  nbins:          number of photon energies
        :type   nbins:          int
        :param  E0:             minimum photon energy [keV]. If None, the minimum of the band
        :type   E0:             unyt.unyt_quantity or None
        :param  E1:             maximum photon energy [keV]. If None, the maximum of the band
        :type   E1:             unyt.unyt_quantity or None
        :param  bin_method:     spacing of the photon energies, 'linear' or 'log'. If None, that of the last
                                spectrum, or 'linear'
        :type   bin_method:     str or None
        """
        E0, E1 = self._energy_bounds(E0, E1)
        if bin_method is None:
            bin_method = "linear" if self.bin_method is None else self.bin_method
        if bin_method=="linear":
            return np.linspace(E0.to_value("keV"), E1.to_value("keV"), nbins) * unyt.keV
        if bin_method=="log":
            if E0<=0:
                raise ValueError("Minimum photon energy must be > 0 for log spaced energies")
            return np.geomspace(E0.to_value("keV"), E1.to_value("keV"), nbins) * unyt.keV
        raise ValueError(f"unknown bin_method '{bin_method}', use 'linear' or 'log'")

    def compute_spectra(self, spectra_model, parameters, nbins, min_energy=None, max_energy=None, bin_method="linear", target_flux=None, norm_flux=None, spec_idx=0, xspec_obj=None, reuse_model=False):
        """
        Compute the spectrum of an AGN point source with pyxspec or a function, with the effective area of the
        telescope. For an xspec model string the bin centres are the nbins photon energies.

        Returns a Spectrum, also kept as self.last_spectrum.

        :param  spectra_model:  the spectrum model, either:
                                    1) a function of the observed photon energy [keV] and 'parameters', e.g.
                                        lambda E, p: (E*(1+p[0]))**-p[1] with p = (redshift, photon index)
                                    2) an xspec model string, e.g. 'zpowerlw'
                                    3) a pyxspec model, using the spectrum number spec_idx
        :type   spectra_model:  callable, str or xspec.Model
        :param  parameters:     parameters of the model:
                                    1) for a function, passed to the function
                                    2) for an xspec model string, a dictionary either numbered by the (1 indexed)
                                        model parameter, e.g. {1: 0.005}, or keyed by component names with the
                                        values of each parameter in order, e.g. {'zpowerlw':[1.8, 0.1, 1]}, or a
                                        dictionary of parameter names, e.g. {'zpowerlw':{'PhoIndex':1.8}}
                                    3) for a pyxspec model, None
        :type   parameters:     any
        :param  nbins:          number of photon energies (or xspec energy bins) to compute the spectrum at
        :type   nbins:          int
        :param  min_energy:     minimum observed photon energy [keV] of the spectrum. If None, the band minimum
        :type   min_energy:     float, unyt.unyt_quantity or None
        :param  max_energy:     maximum observed photon energy [keV] of the spectrum. If None, the band maximum
        :type   max_energy:     float, unyt.unyt_quantity or None
        :param  bin_method:     spacing of the photon energies, 'linear' or 'log'
        :type   bin_method:     str
        :param  target_flux:    (value, units) to scale the spectrum to, over the energy band. The units give a
                                photon ('photon/s/cm**2') or energy ('erg/s/cm**2') flux before the effective area,
                                or after it ('photon/s', 'erg/s'). If None, the spectrum is not scaled
        :type   target_flux:    tuple or None
        :param  norm_flux:      (value, units). If given, the spectrum is scaled by target_flux/norm_flux instead
        :type   norm_flux:      tuple or None
        :param  spec_idx:       spectrum number of a pyxspec model
        :type   spec_idx:       int
        :param  xspec_obj:      PyXspec module, with any additional models and settings (e.g. abundances and
                                cross sections). If None, the imported xspec module
        :type   xspec_obj:      module or None
        :param  reuse_model:    for an xspec model string, if True and the model string and energy grid are the same as
                                the last call with reuse_model=True, the xspec model is kept and only the parameters that
                                changed are set, instead of making a new model. Do not make other xspec models or change
                                the xspec energies between such calls
        :type   reuse_model:    boolean
        """
        if bin_method not in ("linear", "log"):
            raise ValueError(f"unknown bin_method '{bin_method}', use 'linear' or 'log'")
        template_type = self.spectrum_template_type(spectra_model)
        min_energy, max_energy = self._energy_bounds(min_energy, max_energy)

        if template_type=="callable_function":
            x=self.get_observer_energies(nbins, min_energy, max_energy, bin_method)
            y=apply_expected_units(spectra_model(x.to_value("keV"), parameters), SPECTRUM_UNITS)
        else:
            if template_type=="xspec_str":
                xspec_obj = xspec if xspec_obj is None else xspec_obj
                grid = self._xspec_energy_grid(min_energy, max_energy, nbins, bin_method)
                key = (id(xspec_obj), spectra_model, grid)
                cache = _XSPEC_MODEL_CACHE
                if reuse_model and cache["key"] == key:
                    m = cache["model"]
                    cache["params"] = self._set_pyxspec_parameters(m, parameters, previous=cache["params"])
                else:
                    xspec_obj.AllModels.setEnergies(grid)
                    m=xspec_obj.Model(spectra_model)
                    # a new model replaces the active one, so a cached model is no longer valid
                    params_set = self._set_pyxspec_parameters(m, parameters)
                    cache.update(key=key if reuse_model else None, model=m if reuse_model else None, params=params_set)
            else:
                m=spectra_model
            x, y = self._pyxspec_spectrum(m, int(spec_idx), bin_method)

        effective_area = unyt.unyt_array(self.response_function(x.to_value("keV")), units=self._Aeff_units)
        spectrum = Spectrum(x, y, effective_area, bin_method, self.energy_band,
                            model=spectra_model, parameters=parameters, telescope_name=self._telescope_name)

        if target_flux is not None:
            if norm_flux is not None:
                target = target_flux[0] * unyt.Unit(target_flux[1])
                norm = norm_flux[0] * unyt.Unit(norm_flux[1])
                if target.units.dimensions != norm.units.dimensions:
                    raise UnytError("norm and target flux units do no match")
                spectrum = spectrum.scaled((target / norm).to(unyt.dimensionless))
            else:
                spectrum = spectrum.normalised_to(target_flux)

        self.last_spectrum = spectrum
        return spectrum

    def compute_flux(self, min_energy=None, max_energy=None):
        """
        Energy and photon flux of the last spectrum, see Spectrum.flux.

        Returns a tuple of (energy flux [erg/s/cm^2], photon flux [photons/s/cm^2, given as 1/s/cm^2]), each a
        unyt.unyt_quantity (times the effective area units, if the telescope has an effective area).

        :param  min_energy:     lower limit of the integral [keV]. If None, the band minimum
        :type   min_energy:     float, unyt.unyt_quantity or None
        :param  max_energy:     upper limit of the integral [keV]. If None, the band maximum
        :type   max_energy:     float, unyt.unyt_quantity or None
        """
        if self.last_spectrum is None:
            raise ValueError("no spectrum has been computed")
        return self.last_spectrum.flux(min_energy, max_energy)

    @staticmethod
    def basic_template(redshift, nH_factor, spec_index=1.85, norm_fact=1., f_scatter=0., MW_nH_factor=0.018, rel_refl=-1, cos_inc=0.5, fold_energy=300, ratio_abund=1., Fe_abund=1.):
        """
        AGN spectrum with:
            1) intrinsic and Galactic absorption
            2) reflection (pexmon) with the Fe K-alpha line

        Returns a tuple of (xspec model string, parameters dictionary).

        :param  redshift:       redshift of the AGN point source
        :type   redshift:       float
        :param  nH_factor:      intrinsic absorption column density [10^22 cm^-2]
        :type   nH_factor:      float
        :param  spec_index:     photon index of the power law
        :type   spec_index:     float
        :param  norm_fact:      normalisation of the power law
        :type   norm_fact:      float
        :param  f_scatter:      fraction of the emission scattered into the line of sight without intrinsic absorption
        :type   f_scatter:      float
        :param  MW_nH_factor:   Galactic absorption column density [10^22 cm^-2]
        :type   MW_nH_factor:   float
        :param  rel_refl:       relative strength of the reflection, -1 gives only the reflected component
        :type   rel_refl:       float
        :param  cos_inc:        cosine of the AGN inclination angle
        :type   cos_inc:        float
        :param  fold_energy:    cut-off energy of the power law [keV]
        :type   fold_energy:    float
        :param  ratio_abund:    abundance of elements heavier than He relative to solar
        :type   ratio_abund:    float
        :param  Fe_abund:       iron abundance relative to solar
        :type   Fe_abund:       float
        """
        template="tbabs(ztbabs((constant * zpowerlw) + (constant*pexmon)))"
        params={
            "TBabs":{"nH":MW_nH_factor},
            "zTBabs":{"nH":nH_factor, "Redshift":redshift},
            "zpowerlw":_powerlaw_params(redshift, spec_index, norm_fact),
            "pexmon":_pexmon_params(redshift, spec_index, norm_fact, rel_refl, cos_inc, fold_energy, ratio_abund, Fe_abund),
            "constant":{"factor":1-f_scatter},
            "constant_5":{"factor":1-f_scatter},
        }
        return template, params

    @staticmethod
    def AGN_template(redshift, nH_factor, spec_index=1.85, norm_fact=1., f_scatter=0., MW_nH_factor=0.018, rel_refl=-1, cos_inc=0.5, fold_energy=300, ratio_abund=1., Fe_abund=1.):
        """
        AGN spectrum with:
            1) intrinsic and Galactic absorption
            2) reflection (pexmon, ~torus or accretion disc) with the Fe K-alpha line
            3) a fraction f_scatter of the power law scattered by ionised gas without intrinsic absorption

        Returns a tuple of (xspec model string, parameters dictionary).

        :param  redshift:       redshift of the AGN point source
        :type   redshift:       float
        :param  nH_factor:      intrinsic absorption column density [10^22 cm^-2]
        :type   nH_factor:      float
        :param  spec_index:     photon index of the power law
        :type   spec_index:     float
        :param  norm_fact:      normalisation of the power law
        :type   norm_fact:      float
        :param  f_scatter:      fraction of the emission scattered into the line of sight without intrinsic absorption
        :type   f_scatter:      float
        :param  MW_nH_factor:   Galactic absorption column density [10^22 cm^-2]
        :type   MW_nH_factor:   float
        :param  rel_refl:       relative strength of the reflection, -1 gives only the reflected component
        :type   rel_refl:       float
        :param  cos_inc:        cosine of the AGN inclination angle
        :type   cos_inc:        float
        :param  fold_energy:    cut-off energy of the power law [keV]
        :type   fold_energy:    float
        :param  ratio_abund:    abundance of elements heavier than He relative to solar
        :type   ratio_abund:    float
        :param  Fe_abund:       iron abundance relative to solar
        :type   Fe_abund:       float
        """
        template="tbabs(ztbabs((constant * zpowerlw) + (constant*pexmon)) + (constant * zpowerlw))"
        params={
            "TBabs":{"nH":MW_nH_factor},
            "zTBabs":{"nH":nH_factor, "Redshift":redshift},
            "zpowerlw":_powerlaw_params(redshift, spec_index, norm_fact),
            "zpowerlw_8":_powerlaw_params(redshift, spec_index, norm_fact),
            "pexmon":_pexmon_params(redshift, spec_index, norm_fact, rel_refl, cos_inc, fold_energy, ratio_abund, Fe_abund),
            "constant":{"factor":1-f_scatter},
            "constant_5":{"factor":1-f_scatter},
            "constant_7":{"factor":f_scatter},
        }
        return template, params

    @staticmethod
    def SoftExcess_AGN_template(redshift, nH_factor,
        spec_index=1.85, norm_fact=1., f_scatter=0., MW_nH_factor=0.018, rel_refl=-1, cos_inc=0.5, fold_energy=300, ratio_abund=1., Fe_abund=1.,
        plasma_temp=0.2, brem_norm_fact=1.,
        ):
        """
        AGN spectrum with:
            1) intrinsic and Galactic absorption
            2) reflection (pexmon, ~torus or accretion disc) with the Fe K-alpha line
            3) a fraction f_scatter of the power law scattered by ionised gas without intrinsic absorption
            4) a soft excess approximated by bremsstrahlung (zbremss)

        Returns a tuple of (xspec model string, parameters dictionary).

        :param  redshift:       redshift of the AGN point source
        :type   redshift:       float
        :param  nH_factor:      intrinsic absorption column density [10^22 cm^-2]
        :type   nH_factor:      float
        :param  spec_index:     photon index of the power law
        :type   spec_index:     float
        :param  norm_fact:      normalisation of the power law
        :type   norm_fact:      float
        :param  f_scatter:      fraction of the emission scattered into the line of sight without intrinsic absorption
        :type   f_scatter:      float
        :param  MW_nH_factor:   Galactic absorption column density [10^22 cm^-2]
        :type   MW_nH_factor:   float
        :param  rel_refl:       relative strength of the reflection, -1 gives only the reflected component
        :type   rel_refl:       float
        :param  cos_inc:        cosine of the AGN inclination angle
        :type   cos_inc:        float
        :param  fold_energy:    cut-off energy of the power law [keV]
        :type   fold_energy:    float
        :param  ratio_abund:    abundance of elements heavier than He relative to solar
        :type   ratio_abund:    float
        :param  Fe_abund:       iron abundance relative to solar
        :type   Fe_abund:       float
        :param  plasma_temp:    temperature of the bremsstrahlung [keV]
        :type   plasma_temp:    float
        :param  brem_norm_fact: normalisation of the bremsstrahlung
        :type   brem_norm_fact: float
        """
        template="tbabs(ztbabs((constant * zpowerlw) + (constant*pexmon) + (constant*zbremss)) + (constant * zpowerlw))"
        params={
            "TBabs":{"nH":MW_nH_factor},
            "zTBabs":{"nH":nH_factor, "Redshift":redshift},
            "zpowerlw":_powerlaw_params(redshift, spec_index, norm_fact),
            "zpowerlw_10":_powerlaw_params(redshift, spec_index, norm_fact),
            "pexmon":_pexmon_params(redshift, spec_index, norm_fact, rel_refl, cos_inc, fold_energy, ratio_abund, Fe_abund),
            "constant":{"factor":1-f_scatter},
            "constant_5":{"factor":1-f_scatter},
            "constant_7":{"factor":1-f_scatter},
            "zbremss":{"kT":plasma_temp, "Redshift":redshift, "norm":brem_norm_fact},
            "constant_9":{"factor":f_scatter},
        }
        return template, params

    # TODO: templates from published papers (Aird+2015, Seppi+2022, Seppi+2025)


# ---------------------------------------------------------------------------
# numba kernels for the tables
# ---------------------------------------------------------------------------
def get_index_1d(bins, subdata):
    """
    Find the closest bin index below each value, and the relative offset compared to that bin.
    The bins must be evenly spaced.

    Returns a tuple of (bin indices, offsets), each an np.ndarray. Values below the first bin give (0, 0),
    values after the last bin (or NaN) give (len(bins)-2, 1).

    :param  bins:      evenly spaced bin values
    :type   bins:      np.ndarray
    :param  subdata:   values to find the bins of
    :type   subdata:   array-like
    """
    bins = np.ascontiguousarray(bins, dtype=np.float64)
    x = np.ascontiguousarray(subdata, dtype=np.float64)
    idx = np.empty(x.shape, dtype=np.int64)
    dx = np.empty(x.shape, dtype=np.float32)
    _get_index_1d_kernel(bins, x.reshape(-1), idx.reshape(-1), dx.reshape(-1))
    return idx, dx


@njit(cache=True)
def _get_index_1d_kernel(bins, x, idx, dx):
    """
    Fill idx and dx for get_index_1d.

    :param  bins:   evenly spaced bin values
    :type   bins:   np.ndarray
    :param  x:      values to find the bins of, 1D
    :type   x:      np.ndarray
    :param  idx:    output bin indices, same length as x
    :type   idx:    np.ndarray
    :param  dx:     output offsets, same length as x
    :type   dx:     np.ndarray
    """
    eps = 1e-4
    delta = (len(bins) - 1) / (bins[-1] - bins[0])
    for i in range(len(x)):
        if x[i] < bins[0] + eps:
            # below the first element
            idx[i] = 0
            dx[i] = 0
        elif x[i] < bins[-1] - eps:
            # inside the bins
            idx[i] = int((x[i] - bins[0]) * delta)
            dx[i] = (x[i] - bins[idx[i]]) * delta
        else:
            # after the last element (also catches NaN)
            idx[i] = len(bins) - 2
            dx[i] = 1


def _psi(log_Lx, z):
    """
    Absorbed fraction of Compton-thin AGN, psi (Ueda+2014 eq. 3).

    Returns psi as an np.ndarray, in the broadcast shape of the inputs.

    :param  log_Lx:   log10 total intrinsic (de-absorbed) hard (2-10 keV) X-ray luminosity [erg/s]
    :type   log_Lx:   float or np.ndarray
    :param  z:        redshift
    :type   z:        float or np.ndarray
    """
    psi_max = 0.84
    psi_min = 0.2
    a1=0.48
    beta=0.24
    z = np.asarray(z, dtype=np.float64)
    # redshift evolution stops at z=2
    psi_4375 = 0.43 * (1 + np.where(z < 2, z, 2))**a1
    return np.clip(psi_4375 - beta * (np.asarray(log_Lx, dtype=np.float64) - 43.75), psi_min, psi_max)


def _f_NH_from_psi(log_NH, psi):
    """
    Column density probability distribution for given values of psi (Ueda+2014 eq. 4-6).

    Returns the probability of each log_NH as an np.ndarray, in the broadcast shape of the inputs, 0 where psi is NaN.

    :param  log_NH:     log10 intrinsic column density [cm^-2]
    :type   log_NH:     float or np.ndarray
    :param  psi:        absorbed fraction of Compton-thin AGN, from _psi
    :type   psi:        float or np.ndarray
    """
    eps=1.7
    f_CTK=1.0

    log_NH, psi = np.broadcast_arrays(np.asarray(log_NH, dtype=np.float64), np.asarray(psi, dtype=np.float64))
    threshold = (1. + eps) / (3. + eps)
    lower = psi < threshold
    upper = psi >= threshold
    valid = lower | upper # excludes NaN

    # column density ranges
    m1 = (log_NH >= 20) & (log_NH < 21)
    m2 = (log_NH >= 21) & (log_NH < 22)
    m3 = (log_NH >= 22) & (log_NH < 23) & valid
    m4 = (log_NH >= 23) & (log_NH < 24) & valid
    m5 = (log_NH >= 24) & (log_NH < 26) & valid

    result = np.select(
        [m1 & lower, m2 & lower, m1 & upper, m2 & upper, m3, m4, m5],
        [
            1 - ((2 + eps)/(1 + eps) * psi),                  # eq. 5
            (1/(1 + eps)) * psi,
            (2/3) - (((3 + 2*eps)/(3 + 3*eps)) * psi),        # eq. 6
            (1/3) - ((eps/(3 + 3*eps)) * psi),
            (1/(1 + eps)) * psi,
            (eps/(1 + eps)) * psi,
            (f_CTK / 2) * psi,
        ],
        default=0.,
    )

    # apply normalisation
    result[valid] *= 1/(1. + f_CTK * psi[valid])
    return result


@njit(cache=True)
def _splitmix64(x):
    """
    SplitMix64 hash of a 64 bit integer, used to make a random number from a seed and a particle key.

    :param  x:  value to hash
    :type   x:  np.uint64
    """
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@njit(parallel=True, cache=True)
def _sample_nH_kernel(Lbol, z, key1, key2, seed, cdf, nH_values, z_edges, Lx_edges, Lsun_erg_s, c1, c2, k1, k2, out):
    """
    Draw a column density for each particle in one pass: hard X-ray luminosity from the Shen+2020 correction,
    (redshift, luminosity) bin of the table, then the inverse cumulative distribution of that bin at a uniform
    random number made from the seed and the particle's keys.

    Returns the number of particles in bins whose probabilities do not sum to a positive value (given NaN).

    :param  Lbol:           bolometric luminosity of each particle [erg/s]
    :type   Lbol:           np.ndarray
    :param  z:              redshift of each particle
    :type   z:              np.ndarray
    :param  key1:           first 64 bit key of each particle
    :type   key1:           np.ndarray
    :param  key2:           second 64 bit key of each particle
    :type   key2:           np.ndarray
    :param  seed:           random seed
    :type   seed:           np.uint64
    :param  cdf:            cumulative distribution of each bin, shape (redshift bins, luminosity bins, column densities)
    :type   cdf:            np.ndarray
    :param  nH_values:      column densities [cm^-2] of the table
    :type   nH_values:      np.ndarray
    :param  z_edges:        regular redshift bin edges
    :type   z_edges:        np.ndarray
    :param  Lx_edges:       regular log10 hard X-ray luminosity bin edges [erg/s]
    :type   Lx_edges:       np.ndarray
    :param  Lsun_erg_s:     solar luminosity [erg/s]
    :type   Lsun_erg_s:     float
    :param  c1:             Shen+2020 hard band coefficient of the first term
    :type   c1:             float
    :param  c2:             Shen+2020 hard band coefficient of the second term
    :type   c2:             float
    :param  k1:             Shen+2020 hard band power of the first term
    :type   k1:             float
    :param  k2:             Shen+2020 hard band power of the second term
    :type   k2:             float
    :param  out:            output column density of each particle [cm^-2], 0 where Lbol <= 0
    :type   out:            np.ndarray
    """
    inv_z_width = (len(z_edges) - 1) / (z_edges[-1] - z_edges[0])
    inv_Lx_width = (len(Lx_edges) - 1) / (Lx_edges[-1] - Lx_edges[0])
    n_values = cdf.shape[2]
    log_norm = np.log(1e10 * Lsun_erg_s)
    seed_hash = _splitmix64(seed)
    n_bad = 0
    # each particle only depends on itself, so the result does not depend on the number of threads
    for i in prange(len(Lbol)):
        if not (Lbol[i] > 0):
            out[i] = 0.
            continue
        # hard band luminosity, Shen+2020: Lx = Lbol / (c1 q^k1 + c2 q^k2), q = Lbol / 1e10 Lsun
        log_L = np.log(Lbol[i])
        log_q = log_L - log_norm
        log_Lx = (log_L - np.log(c1*np.exp(k1*log_q) + c2*np.exp(k2*log_q))) * 0.4342944819032518 # to log10
        a = _bin_index(z_edges, inv_z_width, z[i])
        b = _bin_index(Lx_edges, inv_Lx_width, log_Lx)
        if not (cdf[a, b, n_values-1] == cdf[a, b, n_values-1]): # NaN, probabilities do not sum to a positive value
            out[i] = np.nan
            n_bad += 1
            continue
        h = _splitmix64(seed_hash ^ _splitmix64(key1[i] ^ _splitmix64(key2[i])))
        uniform = (h >> np.uint64(11)) * (1.0 / 9007199254740992.0) # 53 bits, in [0, 1)
        # first column density with cdf > uniform, as np.searchsorted(side='right')
        lo = 0
        hi = n_values
        while lo < hi:
            mid = (lo + hi) >> 1
            if cdf[a, b, mid] <= uniform:
                lo = mid + 1
            else:
                hi = mid
        out[i] = nH_values[min(lo, n_values-1)]
    return n_bad


def _output_path(output_filename, output_dir=None):
    """
    Path to write a table to, the directory is made if it does not exist.

    Returns the path as a str.

    :param  output_filename:    name of the file, '.hdf5' is appended if missing
    :type   output_filename:    str
    :param  output_dir:         directory of the file. If None, TABLE_DIR, unless output_filename includes a directory
    :type   output_dir:         str or None
    """
    if not output_filename.endswith(".hdf5"):
        output_filename+=".hdf5"
    if output_dir is None:
        output_dir = os.path.dirname(output_filename) or TABLE_DIR
        output_filename = os.path.basename(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output_filename)



@njit(cache=True)
def _bin_index(edges, inv_width, x):
    """
    Index of the regular bin containing x, as in get_index_1d. Values below the first bin give 0, values after
    the last bin (or NaN) give the last bin.

    :param  edges:      regular bin edges
    :type   edges:      np.ndarray
    :param  inv_width:  (number of bins) / (edges[-1] - edges[0])
    :type   inv_width:  float
    :param  x:          value to find the bin of
    :type   x:          float
    """
    eps = 1e-4
    if x < edges[0] + eps:
        return 0
    if x < edges[-1] - eps:
        return int((x - edges[0]) * inv_width)
    return len(edges) - 2


@njit(cache=True)
def _flux_table_kernel(coeffs, fBB, z_mids, z_edges, nH_edges, z, log_nH, out, interpolate, clamp):
    """
    Fill out with the table value of each particle, in one pass over the particles.

    :param  coeffs:         PCHIP coefficients, shape (4, column densities, redshift intervals)
    :type   coeffs:         np.ndarray
    :param  fBB:            table values, shape (column densities, redshifts)
    :type   fBB:            np.ndarray
    :param  z_mids:         regular redshift bin centres
    :type   z_mids:         np.ndarray
    :param  z_edges:        regular redshift bin edges
    :type   z_edges:        np.ndarray
    :param  nH_edges:       regular log10 column density bin edges
    :type   nH_edges:       np.ndarray
    :param  z:              redshift of each particle
    :type   z:              np.ndarray
    :param  log_nH:         log10 column density of each particle
    :type   log_nH:         np.ndarray
    :param  out:            output, one value per particle
    :type   out:            np.ndarray
    :param  interpolate:    If True, interpolate in redshift, else use the nearest redshift bin
    :type   interpolate:    boolean
    :param  clamp:          If True, interpolated redshifts outside of z_mids take the value at the nearest end, else NaN
    :type   clamp:          boolean
    """
    n_int = len(z_mids) - 1
    inv_dz = n_int / (z_mids[-1] - z_mids[0]) if n_int > 0 else 1.
    inv_nH_width = (len(nH_edges) - 1) / (nH_edges[-1] - nH_edges[0])
    inv_z_width = (len(z_edges) - 1) / (z_edges[-1] - z_edges[0])
    for i in range(len(z)):
        k = _bin_index(nH_edges, inv_nH_width, log_nH[i])
        if not interpolate:
            out[i] = fBB[k, _bin_index(z_edges, inv_z_width, z[i])]
            continue

        zi = z[i]
        if zi != zi: # NaN
            out[i] = np.nan
            continue
        if zi < z_mids[0] or zi > z_mids[-1]:
            if not clamp:
                out[i] = np.nan
                continue
            zi = min(max(zi, z_mids[0]), z_mids[-1])
        if n_int == 0:
            out[i] = fBB[k, 0]
            continue
        j = min(int((zi - z_mids[0]) * inv_dz), n_int - 1)
        t = zi - z_mids[j]
        out[i] = ((coeffs[0, k, j]*t + coeffs[1, k, j])*t + coeffs[2, k, j])*t + coeffs[3, k, j]



class FluxTable:
    """
    Broadband flux table of one X-ray band and observation type, loaded once and evaluated for any number of
    particles in a single pass. The PCHIP coefficients in redshift are computed once for every column density.
    """
    def __init__(self, filename, band_name, observation_type, out_of_range="nan"):
        """
        Read the table and compute the interpolation coefficients.

        :param  filename:           path to the table written by PointSourceTables.build_flamingo_broadband_flux_table
        :type   filename:           str
        :param  band_name:          X-ray band, e.g. 'ROSAT'
        :type   band_name:          str
        :param  observation_type:   observation type, e.g. 'photons_convolved'
        :type   observation_type:   str
        :param  out_of_range:       interpolated redshifts outside of the table bin centres give NaN ('nan') or the
                                    value at the nearest bin centre ('clamp'). The nearest bin is always used when
                                    not interpolating
        :type   out_of_range:       str
        """
        if out_of_range not in ("nan", "clamp"):
            raise ValueError(f"unknown out_of_range '{out_of_range}', use 'nan' or 'clamp'")
        self.out_of_range = out_of_range
        self.band_name = band_name
        self.observation_type = observation_type

        with h5py.File(filename, 'r') as f:
            z_edges, z_mids, nH_edges, nH_mids = PointSourceTables._read_table_bins(f["Bins"])
            dset = f[f"{band_name}/{observation_type}/fBB"]
            fBB = dset[()].astype(np.float64)
            self.units = dset.attrs.get("units", None)

        if np.any(z_mids < 0):
            raise ValueError("Bad redshifts in table")
        if fBB.shape != (len(nH_mids), len(z_mids)):
            raise ValueError(f"fBB has shape {fBB.shape}, expected (column densities, redshifts) = {(len(nH_mids), len(z_mids))}")
        for name, values in (("redshift", z_mids), ("log_nH", nH_mids)):
            if len(values) > 2 and not np.allclose(np.diff(values), values[1]-values[0]):
                raise ValueError(f"irregular {name} bins, the table must have regular bins")

        self.redshift = np.ascontiguousarray(z_mids, dtype=np.float64)
        self.log_nH = np.ascontiguousarray(nH_mids, dtype=np.float64)
        self.redshift_edges = np.ascontiguousarray(z_edges, dtype=np.float64)
        self.log_nH_edges = np.ascontiguousarray(nH_edges, dtype=np.float64)
        self.fBB = np.ascontiguousarray(fBB)
        # coefficients c[k, nH bin, z interval] of the cubic in (z - redshift[interval]) for every column density
        if len(self.redshift) > 1:
            self.coeffs = np.ascontiguousarray(np.moveaxis(PchipInterpolator(self.redshift, self.fBB, axis=1).c, 2, 1))
        else:
            self.coeffs = np.zeros((4, len(self.log_nH), 1)) # a single redshift, the kernel uses fBB directly

    def __call__(self, part_redshift, part_log_nH, interpolate=True, out=None, dtype=np.float64):
        """
        Normalised broadband flux of each particle, from its redshift and column density. The nearest column density
        bin is used, and the redshift is either interpolated (PCHIP) or the nearest bin. Call it on chunks of
        particles to limit the memory used.

        Returns the normalised flux of each particle as an np.ndarray (out, if given).

        :param  part_redshift:  redshift of each particle
        :type   part_redshift:  np.ndarray
        :param  part_log_nH:    log10 column density [cm^-2] of each particle
        :type   part_log_nH:    np.ndarray
        :param  interpolate:    If True, interpolate in redshift, else use the nearest redshift bin
        :type   interpolate:    boolean
        :param  out:            array to write the flux into, same length as the particles
        :type   out:            np.ndarray or None
        :param  dtype:          float type of the output if out is None, e.g. np.float32 to halve the memory
        :type   dtype:          numpy dtype
        """
        z = np.ascontiguousarray(part_redshift, dtype=np.float64).reshape(-1)
        log_nH = np.ascontiguousarray(part_log_nH, dtype=np.float64).reshape(-1)
        if len(z) != len(log_nH):
            raise ValueError("part_redshift and part_log_nH must have the same length")
        if out is None:
            out = np.empty(len(z), dtype=dtype)
        elif out.shape != (len(z),) or not out.flags.c_contiguous:
            raise ValueError("out must be a contiguous 1D array with one value per particle")
        _flux_table_kernel(self.coeffs, self.fBB, self.redshift, self.redshift_edges, self.log_nH_edges,
                           z, log_nH, out, interpolate, self.out_of_range=="clamp")
        return out


class PointSourceTables:
    """
    Build, read and sample the tables used for AGN point sources: the column density distribution of
    Ueda+2014, and the broadband flux of AGN spectra as a function of redshift and column density.

    The tables are built and written with the static methods. An instance holds the loaded tables, so they are
    read and prepared once, e.g.

        pst = PointSourceTables(nH_table="nH.hdf5", flux_table="fBB.hdf5", seed=44)
        for chunk in particles:
            nH = pst.sample_nH(chunk.Lbol, chunk.z)
            fBB = pst.flux("ROSAT", "photons_convolved", chunk.z, np.log10(nH))

    For a given seed and nH table, sample_nH gives the same column density to the same particle however the
    particles are ordered or split into chunks, and across instances and runs.
    """
    def __init__(self, nH_table=None, flux_table=None, seed=44, out_of_range="nan", **nH_table_kwargs):
        """
        Set the tables, they are loaded when first used.

        :param  nH_table:           column density probabilities, either a table from make_nH_probability_function_table,
                                    a path to an .hdf5 file written by write_nH_table, or None to make a new table
                                    from nH_table_kwargs
        :type   nH_table:           dict, str or None
        :param  flux_table:         path to a broadband flux table written by build_flamingo_broadband_flux_table
        :type   flux_table:         str or None
        :param  seed:               random seed of sample_nH
        :type   seed:               int
        :param  out_of_range:       redshifts outside of the flux table bin centres give NaN ('nan') or the value at
                                    the nearest bin centre ('clamp'), see FluxTable
        :type   out_of_range:       str
        :param  nH_table_kwargs:    arguments of make_nH_probability_function_table, if nH_table is None
        :type   nH_table_kwargs:    dict
        """
        self.seed = int(seed)
        self.flux_table = flux_table
        self.out_of_range = out_of_range
        self._nH_table_source = nH_table
        self._nH_table_kwargs = nH_table_kwargs
        self._nH = None
        self._flux_tables = {}

    def _load_nH_table(self):
        """
        Load the column density table, once, and compute the cumulative distribution of each (redshift, X-ray
        luminosity) bin, stored with the column densities contiguous in memory.

        Returns a dictionary of the bin edges, column densities and cumulative distributions.
        """
        if self._nH is not None:
            return self._nH
        table = self._nH_table_source
        if table is None:
            table = PointSourceTables.make_nH_probability_function_table(**self._nH_table_kwargs)
        elif isinstance(table, str):
            table = PointSourceTables.read_nH_table(table)
        elif not isinstance(table, dict):
            raise ValueError("did not recognise nH_table")

        if 'log_nH_mids' in table:
            log_nH_mids = np.asarray(table['log_nH_mids'], dtype=np.float64)
        else:
            log_nH_edges = table['log_nH_edges']
            log_nH_mids = np.round(0.5*(log_nH_edges[1:]+log_nH_edges[:-1]), 10)

        # (redshift, column density, luminosity) -> (redshift, luminosity, column density), normalised cumulative sums
        cdf = np.ascontiguousarray(np.transpose(table['probability_function'], (0, 2, 1)), dtype=np.float64)
        np.cumsum(cdf, axis=-1, out=cdf)
        with np.errstate(invalid="ignore", divide="ignore"):
            cdf /= cdf[..., -1:] # bins that do not sum to a positive value become NaN

        self._nH = {
            "z_edges": np.ascontiguousarray(table['z_edges'], dtype=np.float64),
            "log_Lx_edges": np.ascontiguousarray(table['log_Lx_edges'], dtype=np.float64),
            "log_nH_mids": log_nH_mids,
            "nH_values": 10**log_nH_mids,
            "cdf": cdf,
        }
        return self._nH

    def get_flux_table(self, band_name, observation_type):
        """
        Broadband flux table of a band and observation type, loaded from flux_table on first use.

        Returns a FluxTable.

        :param  band_name:          X-ray band, e.g. 'ROSAT'
        :type   band_name:          str
        :param  observation_type:   observation type, e.g. 'photons_convolved'
        :type   observation_type:   str
        """
        if self.flux_table is None:
            raise ValueError("no flux_table was given")
        key = (band_name, observation_type)
        if key not in self._flux_tables:
            self._flux_tables[key] = FluxTable(self.flux_table, band_name, observation_type, out_of_range=self.out_of_range)
        return self._flux_tables[key]

    def flux(self, band_name, observation_type, part_redshift, part_log_nH, interpolate=True, out=None, dtype=np.float64):
        """
        Normalised broadband flux of each particle from the flux table, see FluxTable.

        Returns the normalised flux of each particle as an np.ndarray (out, if given).

        :param  band_name:          X-ray band, e.g. 'ROSAT'
        :type   band_name:          str
        :param  observation_type:   observation type, e.g. 'photons_convolved'
        :type   observation_type:   str
        :param  part_redshift:      redshift of each particle
        :type   part_redshift:      np.ndarray
        :param  part_log_nH:        log10 column density [cm^-2] of each particle
        :type   part_log_nH:        np.ndarray
        :param  interpolate:        If True, interpolate in redshift, else use the nearest redshift bin
        :type   interpolate:        boolean
        :param  out:                array to write the flux into, same length as the particles
        :type   out:                np.ndarray or None
        :param  dtype:              float type of the output if out is None
        :type   dtype:              numpy dtype
        """
        return self.get_flux_table(band_name, observation_type)(part_redshift, part_log_nH, interpolate=interpolate, out=out, dtype=dtype)

    @staticmethod
    def psi_fraction(log_Lx, z):
        """
        Absorbed fraction of Compton-thin AGN, psi (Ueda+2014 eq. 3).

        Returns psi as an np.ndarray, in the broadcast shape of the inputs.

        :param  log_Lx:   log10 total intrinsic (de-absorbed) hard (2-10 keV) X-ray luminosity [erg/s]
        :type   log_Lx:   float or np.ndarray
        :param  z:        redshift
        :type   z:        float or np.ndarray
        """
        return _psi(log_Lx, z)

    @staticmethod
    def f_NH(log_NH, log_Lx, z):
        """
        Column density probability distribution (Ueda+2014 eq. 4-6).

        Returns the probability of each log_NH as an np.ndarray, in the broadcast shape of the inputs.

        :param  log_NH:   log10 intrinsic column density [cm^-2]
        :type   log_NH:   float or np.ndarray
        :param  log_Lx:   log10 total intrinsic (de-absorbed) hard (2-10 keV) X-ray luminosity [erg/s]
        :type   log_Lx:   float or np.ndarray
        :param  z:        redshift
        :type   z:        float or np.ndarray
        """
        return _f_NH_from_psi(log_NH, _psi(log_Lx, z))

    @staticmethod
    def make_nH_probability_function_table(
                log_nH_min=20., log_nH_max=26.,
                log_Lx_min=35., log_Lx_max=55.,
                zmin=0., zmax=6.,
                dnH=0.1, dLx=0.01, dz=0.01,
                output_filename=None, output_dir=None,
                ):
        """
        The column density probability distribution of Ueda+2014 in regular bins of 
        redshift and X-ray luminosity (2-10 keV). 
        The column density and luminosity bins are centred on log_nH_min, ..., log_nH_max 
        and log_Lx_min, ..., log_Lx_max.

        Returns a dictionary of
        - redshift bin edges ('z_edges')
        - log10 X-ray luminosity bin edges ('log_Lx_edges'),
        - log10 column density bin edges and centres ('log_nH_edges', 'log_nH_mids'),
        - the probabilities ('probability_function'),

        :param  log_nH_min:         minimum log10 column density [cm^-2]
        :type   log_nH_min:         float
        :param  log_nH_max:         maximum log10 column density [cm^-2]
        :type   log_nH_max:         float
        :param  log_Lx_min:         minimum log10 X-ray luminosity [erg/s]
        :type   log_Lx_min:         float
        :param  log_Lx_max:         maximum log10 X-ray luminosity [erg/s]
        :type   log_Lx_max:         float
        :param  zmin:               minimum redshift
        :type   zmin:               float
        :param  zmax:               maximum redshift
        :type   zmax:               float
        :param  dnH:                width of the log10 column density bins
        :type   dnH:                float
        :param  dLx:                width of the log10 X-ray luminosity bins
        :type   dLx:                float
        :param  dz:                 width of the redshift bins
        :type   dz:                 float
        :param  output_filename:    name of the file to write the table to. If None, the table is not written
        :type   output_filename:    str or None
        :param  output_dir:         directory to write the table to, see write_nH_table
        :type   output_dir:         str or None
        """
        redshift_bin_edges = _regular_edges(zmin, zmax, dz)
        redshift_bin_mids = 0.5 * (redshift_bin_edges[1:] + redshift_bin_edges[:-1])
        log_Lx_bin_edges, log_Lx_bin_mids = _centred_bins(log_Lx_min, log_Lx_max, dLx)
        log_nH_edges, log_nH_mids = _centred_bins(log_nH_min, log_nH_max, dnH)

        # psi of every (redshift, luminosity) bin, then one redshift at a time to limit temporary memory
        psi = _psi(log_Lx_bin_mids[None, :], redshift_bin_mids[:, None])
        probability = np.empty((len(redshift_bin_mids), len(log_nH_mids), len(log_Lx_bin_mids)))
        for j in range(len(redshift_bin_mids)):
            probability[j] = _f_NH_from_psi(log_nH_mids[:, None], psi[j][None, :])

        table = {
            'z_edges':redshift_bin_edges,
            'log_Lx_edges':log_Lx_bin_edges,
            'log_nH_edges':log_nH_edges,
            'log_nH_mids':log_nH_mids,
            'probability_function':probability,
        }
        if output_filename is not None:
            PointSourceTables.write_nH_table(table, output_filename, output_dir)
        return table

    @staticmethod
    def write_nH_table(table, output_filename, output_dir=None):
        """
        Write a column density probability table to an .hdf5 file.

        Returns the path of the file as a str.

        :param  table:              table from make_nH_probability_function_table
        :type   table:              dict
        :param  output_filename:    name of the file
        :type   output_filename:    str
        :param  output_dir:         directory of the file. If None, TABLE_DIR, unless output_filename includes a directory
        :type   output_dir:         str or None
        """
        path = _output_path(output_filename, output_dir)
        with h5py.File(path, 'w') as f:
            f.create_dataset("DateString", data=datetime.datetime.now().strftime("%d-%B-%Y"))
            for key in ("z_edges", "log_Lx_edges", "log_nH_edges", "log_nH_mids", "probability_function"):
                f.create_dataset(key, data=table[key])
        return path

    @staticmethod
    def read_nH_table(filename):
        """
        Read a column density probability table written by write_nH_table.

        Returns the table as a dictionary, see make_nH_probability_function_table.

        :param  filename:   path to the .hdf5 file
        :type   filename:   str
        """
        with h5py.File(filename, "r") as f:
            table = {key: f[key][()] for key in ("z_edges", "log_Lx_edges", "log_nH_edges", "probability_function")}
            if "log_nH_mids" in f:
                table["log_nH_mids"] = f["log_nH_mids"][()]
        return table


    def sample_nH(self, part_Lbol, part_z, part_ids=None, seed=None, out=None):
        """
        Sample column densities for each particle from the Ueda+2014 column density distribution based on its redshift,
        hard X-ray luminosity bin. 
        Each particle's random number is made from the seed and the particle itself, so for a given seed and nH table 
        the same particle always gets the same column density. 

        By default a particle is identified by its bolometric luminosity [erg/s] and redshift, to a relative precision
        of ~1e-10 (based on KEY_DROPPED_BITS), so values that differ only by rounding (+ unit conversion) are
        the same particle. Alternativley, passin particle ID values can be used instead to identify particles.

        Returns the column density [cm^-2] of each particle as an np.ndarray (out, if given), 0 where Lbol <= 0.

        :param  part_Lbol:  bolometric luminosity of each particle, erg/s if it has no units
        :type   part_Lbol:  unyt.unyt_array or np.ndarray
        :param  part_z:     redshift of each particle
        :type   part_z:     np.ndarray
        :param  part_ids:   integer key of each particle. If None, the luminosity and redshift are used
        :type   part_ids:   np.ndarray or None
        :param  seed:       random seed. If None, self.seed
        :type   seed:       int or None
        :param  out:        array to write the column densities into, same length as the particles
        :type   out:        np.ndarray or None
        """
        if not isinstance(self, PointSourceTables):
            raise TypeError("sample_nH is a method of a PointSourceTables instance, e.g. "
                            "PointSourceTables(nH_table=...).sample_nH(part_Lbol, part_z)")
        table = self._load_nH_table()

        if hasattr(part_Lbol, "units"):
            part_Lbol = part_Lbol.to_value(unyt.erg * unyt.s**-1)
        Lbol = np.ascontiguousarray(part_Lbol, dtype=np.float64).reshape(-1)
        z = np.ascontiguousarray(part_z, dtype=np.float64).reshape(-1)
        if len(Lbol) != len(z):
            raise ValueError("part_Lbol and part_z must have the same length")
        if part_ids is None:
            # The luminosity and redshift identify the particle. 
            # The lowest KEY_DROPPED_BITS of their float64 mantissas are dropped.
            mask = ~np.uint64((1 << KEY_DROPPED_BITS) - 1)
            key1, key2 = Lbol.view(np.uint64) & mask, z.view(np.uint64) & mask
        else:
            key1 = np.ascontiguousarray(part_ids, dtype=np.int64).reshape(-1).view(np.uint64)
            if len(key1) != len(z):
                raise ValueError("part_ids must have one value per particle")
            key2 = key1
        if out is None:
            out = np.empty(len(z), dtype=np.float64)
        elif out.shape != (len(z),) or not out.flags.c_contiguous:
            raise ValueError("out must be a contiguous 1D array with one value per particle")

        seed = self.seed if seed is None else int(seed)
        c1, c2, k1, k2 = SHEN2020_COEFFS["hard"]
        n_bad = _sample_nH_kernel(Lbol, z, key1, key2, np.uint64(seed % 2**64), table["cdf"], table["nH_values"],
                                  table["z_edges"], table["log_Lx_edges"], (1*unyt.Lsun).to_value(unyt.erg/unyt.s),
                                  c1, c2, k1, k2, out)
        if n_bad > 0:
            raise ValueError(f"nH probabilities do not sum to a positive value in the bins of {n_bad} particles")
        return out

    @staticmethod
    def broadband_flux_z(telescope, f_spectra_z, f_norm_flux, E_band_min, E_band_max, numb_energy_bins, redshift_values,
                            ARF_filename=None, ARF_reader=None, xspec_obj=None, progress_bar=False):
        """
        Broadband flux of a normalised spectrum at each redshift. The spectrum is computed at numb_energy_bins+1
        photon energies from the minimum to the maximum of the band, so a power of two uses Romberg integration.

        Returns a tuple of (energy flux, photon flux) at each redshift, each a unyt.unyt_array.

        :param  telescope:          name of the telescope, see AGNSpectra
        :type   telescope:          str
        :param  f_spectra_z:        spectrum model as a function of redshift, returning a tuple of (xspec model
                                    string, parameters)
        :type   f_spectra_z:        callable
        :param  f_norm_flux:        flux to normalise the spectrum by as a function of redshift, returning a tuple of
                                    (value, units), e.g. the analytic flux of the intrinsic power law
        :type   f_norm_flux:        callable
        :param  E_band_min:         minimum observed photon energy of the band [keV]
        :type   E_band_min:         unyt.unyt_quantity
        :param  E_band_max:         maximum observed photon energy of the band [keV]
        :type   E_band_max:         unyt.unyt_quantity
        :param  numb_energy_bins:   number of photon energy intervals across the band
        :type   numb_energy_bins:   int
        :param  redshift_values:    redshifts to compute the flux at
        :type   redshift_values:    np.ndarray
        :param  ARF_filename:       name of the ARF file, see AGNSpectra
        :type   ARF_filename:       str or None
        :param  ARF_reader:         ARF reading function or analytic effective area, see AGNSpectra
        :type   ARF_reader:         callable or None
        :param  xspec_obj:          PyXspec module, with any additional models and settings. If None, the imported xspec module
        :type   xspec_obj:          module or None
        :param  progress_bar:       If True, show a progress bar over the redshifts
        :type   progress_bar:       boolean
        """
        ps = AGNSpectra(telescope, ARF_filename=ARF_filename, ARF_reader=ARF_reader, E_band_min=E_band_min, E_band_max=E_band_max)

        flux_energy = []
        flux_photon = []
        for redshift in tqdm(redshift_values, disable=not progress_bar, desc="sampling redshifts:"):
            model_str, params = f_spectra_z(redshift)
            norm_flux = f_norm_flux(redshift)
            spectrum = ps.compute_spectra(spectra_model=model_str, parameters=params,
                                nbins=numb_energy_bins+1,
                                xspec_obj=xspec_obj, bin_method="linear",
                                target_flux=(1, norm_flux[-1]), norm_flux=norm_flux,
                                )
            fx = spectrum.flux()
            flux_energy.append(fx[0])
            flux_photon.append(fx[1])
        return unyt.unyt_array(flux_energy), unyt.unyt_array(flux_photon)

    @staticmethod
    def table_specs(band_name, observation_type):
        """
        Telescope and energy band of a SWIFT lightcone X-ray band and observation type.

        Returns a tuple of (telescope name, energy band [keV] as a unyt.unyt_array).

        :param  band_name:          X-ray band, a key of FLAMINGO_BANDS, e.g. 'ROSAT' or 'eROSITA-low' (any case)
        :type   band_name:          str
        :param  observation_type:   'photons_intrinsic' (no effective area) or 'photons_convolved'
        :type   observation_type:   str
        """
        band = band_name.lower()
        if band not in FLAMINGO_BANDS:
            raise ValueError(f"band name '{band_name}' not recognised, use one of {list(FLAMINGO_BANDS)}")
        if observation_type == 'photons_intrinsic':
            telescope = "none"
        elif observation_type == 'photons_convolved':
            telescope = "ROSAT" if band == "rosat" else "eROSITA"
        else:
            raise ValueError(f"observation type '{observation_type}' not recognised, use 'photons_intrinsic' or 'photons_convolved'")
        return telescope, unyt.unyt_array(FLAMINGO_BANDS[band], units=unyt.keV)

    @staticmethod
    def build_flamingo_broadband_flux_table(observation_types, observation_bands, output_filename,
                                    log_nH_min=20., log_nH_max=26., dnH=0.1,
                                    zmin=0., zmax=6., dz=0.02,
                                    numb_energy_bins=32768, spec_index=1.85,
                                    xspec_obj=None, f_norm_flux=None, f_spectra_z=None,
                                    output_dir=None, progress_bar=True):
        """
        Build and write a table of the broadband photon flux of an AGN spectrum normalised to an intrinsic flux
        of 1 erg/s/cm^2, as a function of column density and redshift, for each X-ray band and observation type.

        The file has the bins in 'Bins' ('z_edges', 'log_nH_edges' and their centres 'redshift', 'log_nH'), the xspec
        model and its parameters as the attributes of 'spectrum' (see _spectrum_attributes), and the flux of each band
        and observation type in '<band>/<observation type>/fBB', shape (column densities, redshifts).

        Returns the path of the file as a str.

        :param  observation_types:  observation types, see table_specs
        :type   observation_types:  list of str
        :param  observation_bands:  X-ray bands, see table_specs
        :type   observation_bands:  list of str
        :param  output_filename:    name of the file
        :type   output_filename:    str
        :param  log_nH_min:         minimum log10 column density [cm^-2]
        :type   log_nH_min:         float
        :param  log_nH_max:         maximum log10 column density [cm^-2]
        :type   log_nH_max:         float
        :param  dnH:                width of the log10 column density bins
        :type   dnH:                float
        :param  zmin:               minimum redshift
        :type   zmin:               float
        :param  zmax:               maximum redshift
        :type   zmax:               float
        :param  dz:                 width of the redshift bins
        :type   dz:                 float
        :param  numb_energy_bins:   number of photon energy intervals across the band, see broadband_flux_z
        :type   numb_energy_bins:   int
        :param  spec_index:         photon index of the default spectrum and normalisation
        :type   spec_index:         float
        :param  xspec_obj:          PyXspec module, with any additional models and settings. If None, the imported xspec module
        :type   xspec_obj:          module or None
        :param  f_norm_flux:        flux to normalise by, a function of (redshift, minimum and maximum band energy [keV])
                                    returning (value, units). If None, the analytic energy flux of a power law
                                    with photon index spec_index. The fBB attribute 'NormPhotonIndex' is spec_index
                                    if f_norm_flux is None, otherwise -1
        :type   f_norm_flux:        callable or None
        :param  f_spectra_z:        spectrum model, a function of (redshift, column density [10^22 cm^-2]) returning
                                    (xspec model string, parameters). If None, AGNSpectra.basic_template with spec_index
        :type   f_spectra_z:        callable or None
        :param  output_dir:         directory of the file. If None, TABLE_DIR, unless output_filename includes a directory
        :type   output_dir:         str or None
        :param  progress_bar:       If True, show a progress bar over the redshifts
        :type   progress_bar:       boolean
        """
        for xtype in observation_types:
            if "energies" in xtype:
                raise ValueError(f"{xtype} has not been included yet")

        if f_spectra_z is None:
            f_spectra_z = lambda z, nH_factor: AGNSpectra.basic_template(z, nH_factor, spec_index=spec_index)
        # photon index of the power law the fluxes are normalised by, -1 if f_norm_flux is given (not known)
        norm_photon_index = float(spec_index) if f_norm_flux is None else -1.
        if f_norm_flux is None:
            f_norm_flux = lambda z, E_min, E_max: (
                (AGNSpectra.analytic_plaw_flux_energy(E_min.to_value("keV"), E_max.to_value("keV"), z, spec_index) * unyt.keV).to_value(unyt.erg),
                "erg/s/cm**2")

        log_nH_edges, log_nH_values = _centred_bins(log_nH_min, log_nH_max, dnH)
        redshift_edges = _regular_edges(zmin, zmax, dz)
        redshift_values = 0.5 * (redshift_edges[1:] + redshift_edges[:-1])
        nH_factors = [10**log_nH / 1e22 for log_nH in log_nH_values]

        path = _output_path(output_filename, output_dir)
        with h5py.File(path, 'w') as outfile:
            outfile.create_dataset("DateString", data=datetime.datetime.now().strftime("%d-%B-%Y"))
            bin_data = outfile.create_group("Bins")
            bin_data.create_dataset("z_edges", data=redshift_edges)
            bin_data.create_dataset("log_nH_edges", data=log_nH_edges)
            bin_data.create_dataset("redshift", data=redshift_values)
            bin_data.create_dataset("log_nH", data=log_nH_values)
            bin_data.create_dataset("observation_bands", data=np.array(observation_bands, dtype=h5py.string_dtype()))
            bin_data.create_dataset("observation_types", data=np.array(observation_types, dtype=h5py.string_dtype()))

            # xspec model and parameters of the spectra
            spectrum_data = outfile.create_group("spectrum")
            for name, value in PointSourceTables._spectrum_attributes(f_spectra_z, redshift_values, nH_factors).items():
                spectrum_data.attrs[name] = value

            for xband in observation_bands:
                band_data = outfile.require_group(xband)
                for xtype in observation_types:
                    telescope_name, Eband_range = PointSourceTables.table_specs(xband, xtype)
                    print(f"\nMaking Table of {xband} band {xtype} values\n", flush=True)
                    print(f"\tsampling spectra with {numb_energy_bins} datapoints", flush=True)

                    ps = AGNSpectra(telescope_name, E_band_min=Eband_range[0], E_band_max=Eband_range[1])
                    output_data_fBB = np.zeros((len(log_nH_values), len(redshift_values)), dtype=float)
                    # redshift outer, column density inner: within a redshift only the absorption changes, so one
                    # xspec model is kept and only its changed parameters are set, rather than making a new model
                    # (and recomputing every component, e.g. pexmon) for every spectrum
                    for j, z in tqdm(enumerate(redshift_values), total=len(redshift_values), disable=not progress_bar):
                        norm_flux = f_norm_flux(z, Eband_range[0], Eband_range[1])
                        for i, nH_factor in enumerate(nH_factors):
                            model_str, params = f_spectra_z(z, nH_factor)
                            spectrum = ps.compute_spectra(model_str, params, numb_energy_bins+1,
                                                          xspec_obj=xspec_obj, bin_method="linear",
                                                          target_flux=(1, norm_flux[-1]), norm_flux=norm_flux,
                                                          reuse_model=True)
                            fBB = spectrum.flux()[1]
                            output_data_fBB[i, j] = fBB.value

                    obvs_data = band_data.create_group(xtype)
                    dset = obvs_data.create_dataset('fBB', data=output_data_fBB, dtype=float)
                    dset.attrs["units"] = str(fBB.units)
                    dset.attrs["normalisation"] = "photon flux of the spectrum divided by f_norm_flux, by default an intrinsic power law energy flux of 1 erg/s/cm**2"
                    dset.attrs["telescope"] = telescope_name
                    dset.attrs["energy_band_keV"] = Eband_range.to_value("keV")
                    dset.attrs["NormPhotonIndex"] = norm_photon_index
        return path

    @staticmethod
    def _spectrum_attributes(f_spectra_z, redshift_values, nH_factors):
        """
        xspec model and parameters used to make a broadband flux table, as attributes for its 'spectrum' group.
        'model' is the xspec model string. Each parameter is named '<component>.<parameter>' (or 'par<number>' for
        numbered parameters) and is a single value if it is the same across the table. Parameters that change across
        the table (the redshifts and column densities) are given as their [minimum, maximum], found from the spectra at
        the corners of the table rather than every point on the grid.

        Returns the attributes as a dictionary.

        :param  f_spectra_z:        spectrum model, a function of (redshift, column density [10^22 cm^-2]) returning
                                    (xspec model string, parameters), see build_flamingo_broadband_flux_table
        :type   f_spectra_z:        callable
        :param  redshift_values:    redshifts of the table
        :type   redshift_values:    np.ndarray
        :param  nH_factors:         column densities of the table [10^22 cm^-2]
        :type   nH_factors:         list of float
        """
        models = []
        values = {}
        for z in (redshift_values[0], redshift_values[-1]):
            for nH_factor in (nH_factors[0], nH_factors[-1]):
                model_str, params = f_spectra_z(z, nH_factor)
                if model_str not in models:
                    models.append(model_str)
                for comp_name, comp_params in (params or {}).items():
                    if isinstance(comp_params, dict):
                        items = [(f"{comp_name}.{par_name}", value) for par_name, value in comp_params.items()]
                    else:
                        items = [(f"par{comp_name}", comp_params)]
                    for name, value in items:
                        values.setdefault(name, [])
                        if value not in values[name]:
                            values[name].append(value)

        if len(models) != 1:
            raise ValueError(f"the xspec model changes across the table: {models}")
        attrs = {"model": models[0]}
        for name, vals in values.items():
            if len(vals) == 1:
                attrs[name] = vals[0]
            else:
                vals = np.asarray(vals, dtype=float)
                attrs[name] = np.array([vals.min(), vals.max()])
        return attrs

    @staticmethod
    def _read_table_bins(bins):
        """
        Redshift and log10 column density bin edges and centres of a broadband flux table. Tables with only
        the bin centres ('redshift', 'log_nH') are also read, the edges are then assumed to be regular.

        Returns a tuple of (redshift edges, redshift centres, log10 nH edges, log10 nH centres), each an np.ndarray.

        :param  bins:   'Bins' group of the table
        :type   bins:   h5py.Group
        """
        def edges_and_mids(edges_name, mids_name):
            """
            Returns a tuple of (bin edges, bin centres) of one table axis, each an np.ndarray.

            :param  edges_name: dataset name of the bin edges in 'Bins'
            :type   edges_name: str
            :param  mids_name:  dataset name of the bin centres in 'Bins'
            :type   mids_name:  str
            """
            if mids_name in bins:
                mids = bins[mids_name][()]
            else:
                edges = bins[edges_name][()]
                mids = 0.5*(edges[1:]+edges[:-1])
            if edges_name in bins:
                edges = bins[edges_name][()]
            else:
                width = mids[1]-mids[0] if len(mids)>1 else 2.
                edges = np.append(mids-width/2, mids[-1]+width/2)
            return edges, mids
        redshift_edges, redshift_mids = edges_and_mids("z_edges", "redshift")
        log_nH_edges, log_nH_mids = edges_and_mids("log_nH_edges", "log_nH")
        return redshift_edges, redshift_mids, log_nH_edges, log_nH_mids

    @staticmethod
    def load_flux_table(filename, band_name, observation_type, out_of_range="nan"):
        """
        Load a broadband flux table once, to evaluate it for many particles or chunks of particles.

        Returns a FluxTable.

        :param  filename:           path to the table written by build_flamingo_broadband_flux_table
        :type   filename:           str
        :param  band_name:          X-ray band, e.g. 'ROSAT'
        :type   band_name:          str
        :param  observation_type:   observation type, e.g. 'photons_convolved'
        :type   observation_type:   str
        :param  out_of_range:       redshifts outside of the table bin centres give NaN ('nan') or the value at the
                                    nearest bin centre ('clamp'), see FluxTable
        :type   out_of_range:       str
        """
        return FluxTable(filename, band_name, observation_type, out_of_range=out_of_range)

    @staticmethod
    def normalised_broadband_flux_from_table(filename,
                observation_type, band_name, part_redshift, part_log_nH,
                interpolate_table=True,
                quiet=False):
        """
        Normalised broadband flux of each particle, from its redshift and column density. The nearest column
        density bin is used, and the redshift is either interpolated (PCHIP) or the nearest bin. Loads the table
        on every call, use load_flux_table to evaluate the same table many times.

        Returns the normalised flux of each particle as an np.ndarray, NaN for interpolated redshifts outside of
        the table bin centres.

        :param  filename:           path to the table written by build_flamingo_broadband_flux_table
        :type   filename:           str
        :param  observation_type:   observation type, e.g. 'photons_convolved'
        :type   observation_type:   str
        :param  band_name:          X-ray band, e.g. 'ROSAT'
        :type   band_name:          str
        :param  part_redshift:      redshift of each particle
        :type   part_redshift:      np.ndarray
        :param  part_log_nH:        log10 column density [cm^-2] of each particle
        :type   part_log_nH:        np.ndarray
        :param  interpolate_table:  If True, interpolate in redshift, else use the nearest redshift bin
        :type   interpolate_table:  boolean
        :param  quiet:              If False, print the range of the particles and of the table
        :type   quiet:              boolean
        """
        table = FluxTable(filename, band_name, observation_type)
        if quiet is False:
            print(f"Reading {band_name}/{observation_type} for particles with\nz:\t({np.min(part_redshift)}, {np.max(part_redshift)})"
                  f"\nlog_nH:\t({np.min(part_log_nH)}, {np.max(part_log_nH)})\nfrom a table covering\nz:\t({table.redshift[0]}, {table.redshift[-1]})"
                  f"\nlog_nH:\t({table.log_nH[0]}, {table.log_nH[-1]})", flush=True)
        return table(part_redshift, part_log_nH, interpolate=interpolate_table)



