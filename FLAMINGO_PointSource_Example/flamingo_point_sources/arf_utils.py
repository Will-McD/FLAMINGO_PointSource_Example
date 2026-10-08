#!/bin/env python
"""
Reading telescope ancillary response files (ARFs) 
The ARF files for eROSITA (+ all added telescopes) are looked for in the ARF directory at the top of the repository. 
The response function for ROSAT has been tabulated and hard coded in. 
"""
import os
import h5py
import numpy as np
from astropy.io import fits

# directory the ARF files are kept in, ARF/ at the top of the repository
ARF_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ARF")

# ARF files of the preset telescopes, in ARF_dir
PRESET_ARF_FILES = {
    "eROSITA": "survey_tm8_arf_filter_2023-01-17.fits",
}


def read_fits_specresp(filename):
    """
    Read the SPECRESP table of an OGIP ARF fits file.

    Returns a tuple of (energy bin centres [keV], effective areas [cm^2]) as np.ndarrays.

    :param  filename:   path to the ARF fits file
    :type   filename:   str
    """
    with fits.open(filename) as hdul:
        # columns ENERG_LO, ENERG_HI [keV] and SPECRESP [cm^2], read by position
        specresp = hdul["SPECRESP"].data
        energ_lo = np.array(specresp.field(0), dtype=float)
        energ_hi = np.array(specresp.field(1), dtype=float)
        effective_areas = np.array(specresp.field(2), dtype=float)
    return (energ_lo + energ_hi) / 2, effective_areas


def eROSITA_response_function(filename):
    """
    Read the eROSITA ARF.

    Returns a tuple of (energy bin centres [keV], effective areas [cm^2]) as np.ndarrays.

    :param  filename:   path to the eROSITA ARF fits file
    :type   filename:   str
    """
    return read_fits_specresp(filename)


def ROSAT_response_function():
    """
    ROSAT effective area.

    Returns a tuple of (energies [keV], effective areas [cm^2]) as np.ndarrays.
    """
    energy_centres = np.array([
        0.10, 0.20, 0.279, 0.28, 0.30, 0.40, 0.50, 0.60, 0.63, 0.70, 0.80, 0.90, 1.00,
        1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.8, 1.90, 2.00, 2.10, 2.20, 2.30, 3.40
    ])
    effective_areas = np.array([
        1.0, 20.2, 43.5, 0.0, 1.0, 10.6, 19.3, 7.6, 11.9, 35.5, 59.9, 82.6, 83.2,
        94.5, 89.7, 84.7, 76.1, 56.5, 50.0, 37.2, 28.0, 19.4, 11.2, 2.7, 0.6, 0.4, 0.2
    ])
    return energy_centres, effective_areas


def get_ARF_path(arf_filename):
    """
    Path to an ARF file in ARF_dir, checking that it exists.

    Returns the path as a str, or None for telescopes without an ARF file ('ROSAT' and 'none').

    :param  arf_filename:   name of the file in ARF_dir, or a preset telescope name ('eROSITA', 'ROSAT', 'none')
    :type   arf_filename:   str
    """
    if arf_filename in ("ROSAT", "none"):
        return None
    arf_filename = PRESET_ARF_FILES.get(arf_filename, arf_filename)

    path = os.path.join(ARF_dir, arf_filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ARF file not found: {path}")
    return path


def _read_ARF_by_extension(filename):
    """
    Read an ARF file with a layout guessed from its extension: 
    - .hdf5 with 'Energy' and 'Area' datasets. 
    - .txt or .csv, assumes comma seperated columns of energy and area columns
    - .fits, a SPECRESP table.

    Returns a tuple of (energies [keV], effective areas [cm^2]) as np.ndarrays.

    :param  filename:   path to the ARF file
    :type   filename:   str
    """
    if filename.endswith((".hdf5", ".h5")):
        with h5py.File(filename, "r") as f:
            missing = [name for name in ("Energy", "Area") if name not in f]
            if missing:
                raise KeyError(f"ARF file {filename} has no {' or '.join(missing)} dataset")
            return f["Energy"][()], f["Area"][()]
    if filename.endswith(".txt"):
        table = np.loadtxt(filename, delimiter=",")
        return table[:, 0], table[:, 1]
    if filename.endswith(".fits"):
        return read_fits_specresp(filename)
    raise ValueError(f"ARF file not recognised: {filename}, give a reader_function")


def read_ARF(arf_filename, reader_function=None):
    """
    Read an ARF from ARF_dir, or a preset telescope's response function.

    Returns a tuple of (energies [keV], effective areas [cm^2]) as np.ndarrays.

    :param  arf_filename:       name of the file in ARF_dir, or a preset telescope name ('ROSAT' or 'eROSITA')
    :type   arf_filename:       str
    :param  reader_function:    function taking the path to the ARF file and returning (energies, effective areas).
                                Only used for files that are not presets. If None, the layout is guessed from the
                                file extension (.hdf5/.h5, .txt or .fits)
    :type   reader_function:    callable or None
    """
    if arf_filename == "ROSAT":
        return ROSAT_response_function()
    if arf_filename == "none":
        raise ValueError("telescope 'none' has no ARF, its effective area is 1")

    filename = get_ARF_path(arf_filename)
    if arf_filename == "eROSITA":
        return eROSITA_response_function(filename)
    if reader_function is not None:
        return reader_function(filename)
    return _read_ARF_by_extension(filename)