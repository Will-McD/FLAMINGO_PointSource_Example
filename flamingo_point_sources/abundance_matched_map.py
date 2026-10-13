#!/bin/env python
"""
Build HEALPix maps of X-ray emission from AGN point sources in the FLAMINGO lightcone 
shells by abundance matching to the Shen+2020 quasar luminosity function (QLF).

For a QLF model at a given redshift, the haloes in the redshift range it covers are 
read from the halo lightcone and SOAP, bolometric luminosities are sampled from the QLF 
for the comoving volume of the range and assigned to the haloes in rank order 
(by black hole accretion rate, or by Vmax with log-normal scatter). 
Each AGN gets a column density, its X-ray flux is found with the Shen+2020 bolometric correction and 
its count rate with the broadband flux table. 
The fluxes and count rates are added to the HEALPix map of each lightcone shell.
"""
import os
import io
import argparse
import contextlib
import h5py
import numpy as np
import healpy as hp
import unyt
from swiftlet.swift_snapshot_redshift_conversion import snapshot_number_in_range, flamingo_shell_redshift_file, flamingo_box_resolution
from swiftlet.healpix_map_utils import read_rotation_angles
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone
from lightcone_io.smoothed_map import rotate_particle_coordinates
from .sample_qlf import SampleQLF, Shen_2020_bolometric_correation
from .agn_spectra import PointSourceTables

import virgo.mpi.parallel_hdf5 as phdf5


# X-ray bands with a known bolometric correction: the Shen+2020 soft band (0.5-2 keV) is the ROSAT band
BOLOMETRIC_CORRECTION_BANDS = {"rosat": "soft"}

# number of particles placed on the maps at a time
CHUNK_SIZE = 65536


def report(message, quiet=False, **print_kwargs):
    """
    Print a progress message, unless quiet is True.

    :param  message:        message to print
    :type   message:        str
    :param  quiet:          If True, print nothing
    :type   quiet:          boolean
    :param  print_kwargs:   keyword arguments of print, e.g. end='\\r'
    :type   print_kwargs:   dict
    """
    if not quiet:
        print(message, flush=True, **print_kwargs)


def shell_redshift_box(box_res):
    """
    Returns the FLAMINGO shell redshift file label ('L1' or 'L2p8') of a simulation box, see swiftlet's
    flamingo_shell_redshift_file.

    :param  box_res:    FLAMINGO box size and resolution label, e.g. "L1000N1800"
    :type   box_res:    str
    """
    if box_res.startswith("L1000"):
        return "L1"
    elif box_res.startswith("L2800"):
        return "L2p8"
    raise ValueError(f"no lightcone shell redshift file for box {box_res}")


def assign_lightcone_shell(particle_z, shell_box):
    """
    Returns the lightcone shell number of each particle, from its redshift and the FLAMINGO shell redshifts.

    :param  particle_z:     redshift of each particle
    :type   particle_z:     np.ndarray
    :param  shell_box:      shell redshift file label, 'L1' or 'L2p8', see shell_redshift_box
    :type   shell_box:      str
    """
    shell_edges = np.loadtxt(flamingo_shell_redshift_file(shell_box), delimiter=",")
    return np.digitize(particle_z, bins=shell_edges[:, -1], right=False)


def read_particle_data(halo_format, soap_format, snapshot_numbers, qlf_redshift_range, shell_box, comm=None, quiet=False):
    """
    Read the halo lightcone and corresponding SOAP catalogues over a given redshift range.
    Haloes beyond the last lightcone shell of the shell redshift file are left out.

    Returns a dictionary of np.ndarray: 
    - halo lightcone properties:
        ('Lightcone/Redshift', 'Lightcone/HaloCentre', 'InputHalos/SOAPIndex', 'Lightcone/SnapshotNumber')
        
    - computed properties:
        'Lightcone/ShellNumber'
    
    - SOAP properties: 
        ('BoundSubhalo/MaximumCircularVelocity', 'BoundSubhalo/MostMassiveBlackHoleAccretionRate' [g/s, physical CGS], 'BoundSubhalo/MostMassiveBlackHoleID', 'InputHalos/HaloCatalogueIndex')

    :param  halo_format:        halo lightcone file names, with the snapshot number as {file_nr:04d}
    :type   halo_format:        str
    :param  soap_format:        SOAP file names, with the snapshot number as {snap_nr:04}
    :type   soap_format:        str
    :param  snapshot_numbers:   snapshots to read
    :type   snapshot_numbers:   np.ndarray
    :param  qlf_redshift_range: (minimum, maximum) redshift of the haloes to keep
    :type   qlf_redshift_range: tuple
    :param  shell_box:          shell redshift file label, 'L1' or 'L2p8', see shell_redshift_box
    :type   shell_box:          str
    :param  comm:               MPI communicator for virgo's MultiFile. If None, MPI.COMM_WORLD
    :type   comm:               mpi4py.MPI.Comm or None
    :param  quiet:              If True, do not print progress
    :type   quiet:              boolean
    """
    zmin, zmax = qlf_redshift_range[0], qlf_redshift_range[1]
    report("\nLoading haloes....", quiet)

    mf = phdf5.MultiFile(halo_format, file_idx=np.asarray(snapshot_numbers, dtype=int), comm=comm)
    halo_properties = ('Lightcone/Redshift', 'Lightcone/HaloCentre', 'InputHalos/SOAPIndex', 'Lightcone/SnapshotNumber')
    halo_data = mf.read(halo_properties, group="/", read_attributes=True)

    in_range = (halo_data['Lightcone/Redshift'] > zmin) & (halo_data['Lightcone/Redshift'] <= zmax)
    shell_number = assign_lightcone_shell(halo_data['Lightcone/Redshift'], shell_box)
    # haloes beyond the last lightcone shell cannot be mapped
    n_shells = len(np.loadtxt(flamingo_shell_redshift_file(shell_box), delimiter=","))
    beyond_shells = in_range & (shell_number >= n_shells)
    if np.any(beyond_shells):
        report(f"\t{np.sum(beyond_shells)} haloes beyond the last lightcone shell are not used", quiet)
    in_range &= ~beyond_shells

    particle_data = {}
    for prop_name in halo_properties:
        report(f"\treading {prop_name} values for {np.sum(in_range)} haloes", quiet)

        particle_data[prop_name] = np.asarray(halo_data[prop_name][in_range], dtype=halo_data[prop_name].dtype)
    particle_data["Lightcone/ShellNumber"] = shell_number[in_range]
    n_haloes = len(particle_data['Lightcone/SnapshotNumber'])

    soap_properties = ('BoundSubhalo/MaximumCircularVelocity', 'BoundSubhalo/MostMassiveBlackHoleAccretionRate',
                       'BoundSubhalo/MostMassiveBlackHoleID', 'InputHalos/HaloCatalogueIndex')
    for snap_nr in snapshot_numbers:
        in_snapshot = particle_data['Lightcone/SnapshotNumber'] == snap_nr
        if not np.any(in_snapshot):
            continue
        report(f"\tadding soap properties for snapshot {snap_nr}...", quiet)
        soap_idx = particle_data['InputHalos/SOAPIndex'][in_snapshot].astype(np.int64)
        
        # h5py reads need increasing, unique indices
        unique_soap_idx, soap_order = np.unique(soap_idx, return_inverse=True)

        with h5py.File(soap_format.format(snap_nr=snap_nr), 'r') as infile:
            for prop_name in soap_properties:
                dset = infile[prop_name]
                if prop_name not in particle_data:
                    dtype = np.float64 if prop_name == 'BoundSubhalo/MostMassiveBlackHoleAccretionRate' else dset.dtype
                    particle_data[prop_name] = np.zeros((n_haloes,)+dset.shape[1:], dtype=dtype)

                values = dset[unique_soap_idx][soap_order]
                if prop_name == 'BoundSubhalo/MostMassiveBlackHoleAccretionRate':
                    # physical CGS in float64, the internal units times the conversion factor can lose precision in float32
                    values = values.astype(np.float64) * dset.attrs['Conversion factor to physical CGS (including cosmological corrections)']
                particle_data[prop_name][in_snapshot] = values

    return particle_data


def get_particle_data_for_abundance_matching(args):
    """
    Read the haloes in the selected QLF models redshift range.
    The QLF redshift range is defined by the midpoint between the neighbouring QLF redshifts. 

    Returns a tuple: 
        (halo data, see read_particle_data, (minimum, maximum) redshift, tabulated QLF redshift, cosmology as a lightcone_io Snapshot_Cosmology_For_Lightcone, shell redshift file label).

    :param  args:   command line arguments, see the end of this file
    :type   args:   argparse.Namespace
    """
    qlf = SampleQLF(args.qlf_filename)
    qlf_redshifts = qlf.qlf_params["z"]
    qlf_redshift = qlf_redshifts[args.qlf_shell_nr]
    if args.qlf_shell_nr == len(qlf_redshifts)-1:
        z_max = qlf_redshift
    else:
        z_max = qlf_redshift + 0.5 * (qlf_redshifts[args.qlf_shell_nr+1] - qlf_redshifts[args.qlf_shell_nr])
    if args.qlf_shell_nr == 0 or qlf_redshift <= 0.21:
        z_min = 0
    else:
        z_min = qlf_redshift - 0.5 * (qlf_redshifts[args.qlf_shell_nr] - qlf_redshifts[args.qlf_shell_nr-1])
    del qlf
    z_range = (z_min, z_max)

    # snapshots overlapping the redshift range, and one either side
    snapshots_to_join = snapshot_number_in_range(
                redshift_range=z_range, boxsize_resolution=args.box_res,
                redshift_buffer=(0., 0.), snapshot_buffer=(1, 1),
                bounds="equal", decimals=5)

    report(f"Using Box: {args.box_res}\nSimulation: {args.sim_name}\n", args.quiet)

    simulation_dir = f"/cosma8/data/dp004/flamingo/Runs/{args.box_res}/{args.sim_name}"
    halo_lightcone_dir = f"{simulation_dir}/sorted_hbt_lightcone_halos/lightcone{args.lightcone_nr}"
    if os.path.exists(halo_lightcone_dir) is False:
        # try back-up option
        second_dir = f"{simulation_dir}/hbt_lightcone_halos/lightcone{args.lightcone_nr}"
        use_second_dir = os.path.exists(second_dir)
        if use_second_dir:
            halo_lightcone_dir=second_dir
        else:
            raise ValueError(f"no halo lightcone directory known for ./{args.box_res}/{args.sim_name}")

    halo_format = halo_lightcone_dir+"/lightcone_halos_{file_nr:04d}.hdf5" # use file_nr for virgodc multifile 
    soap_format = f'{simulation_dir}/SOAP-HBT/halo_properties_'+'{snap_nr:04}'+'.hdf5'
    shell_box = shell_redshift_box(flamingo_box_resolution(simulation_dir))
    cosmo = Snapshot_Cosmology_For_Lightcone(simulation_dir+"/snapshots")

    particle_data = read_particle_data(halo_format, soap_format, snapshots_to_join, z_range, shell_box, quiet=args.quiet)

    report(f"\n\tredshift range of particles: {particle_data['Lightcone/Redshift'].min()},  {particle_data['Lightcone/Redshift'].max()}", args.quiet)
    report(f"\tnumb particles read: {len(particle_data['Lightcone/Redshift']):.3e}", args.quiet)

    return particle_data, z_range, qlf_redshift, cosmo, shell_box


def sample_luminosities(args, number_of_BH, qlf_redshift_range, midplane_redshift, cosmo):
    """
    Sample bolometric luminosities from the QLF model for the comoving volume of the lightcone for the given redshift range

    Returns the luminosities as a unyt.unyt_array [erg/s], brightest first, zero for any halo beyond the number
    of luminosities sampled.

    :param  args:               command line arguments, see the end of this file. Log_Lmin is raised to 30 if lower
    :type   args:               argparse.Namespace
    :param  number_of_BH:       number of haloes
    :type   number_of_BH:       int
    :param  qlf_redshift_range: (minimum, maximum) redshift of the haloes
    :type   qlf_redshift_range: tuple
    :param  midplane_redshift:  redshift selecting the QLF and seeding its sample, see SampleQLF.sample_LF_Lbol
    :type   midplane_redshift:  float
    :param  cosmo:              cosmology of the simulation
    :type   cosmo:              lightcone_io.xray_utils.Snapshot_Cosmology_For_Lightcone
    """
    report("\nPerforming abundance matching ....", args.quiet)

    if args.Log_Lmin <= 30:
        args.Log_Lmin = 30
    report(f"\tAllowed minimum bolometric luminosity: {10**args.Log_Lmin:.2e} [erg/s]", args.quiet)
    qlf = SampleQLF(args.qlf_filename)
    report(f"\tQLF redshift: {qlf.get_nearest_LF_z_model(midplane_redshift):.2f}\n", args.quiet)

    lightcone_volume = cosmo.lightcone_volume(qlf_redshift_range[0], qlf_redshift_range[-1], use_redshift=True, phi=None)
    sampled_Lbol, running_total = qlf.sample_LF_Lbol(
                        midplane_redshift, number_of_BH, lightcone_volume,
                        log_Lbol_bin_size=0.1,
                        log_Lbol_min_value=args.Log_Lmin,
                        log_Lbol_max_value=50,
                        min_expected=1.,
                        progress_bar=False,
                        per_shell_seed=True,
                        within_bin="qlf",
                        exact_bins=True,
                        bright_end="all"
                        )
    assert len(sampled_Lbol) == running_total
    number_of_L_sampled = len(sampled_Lbol)
    report(f"\n\tnumb luminosities sampled: {number_of_L_sampled:.3e}\n\tnumb BH's read: {number_of_BH:.3e}", args.quiet)

    # the sample is brightest first, keep one per halo
    ordered_sampled_Lbol = unyt.unyt_array(np.zeros(number_of_BH, dtype=np.float64), units=unyt.erg/unyt.s)
    n_keep = min(number_of_L_sampled, number_of_BH)
    ordered_sampled_Lbol[0:n_keep] += sampled_Lbol[0:n_keep].to(unyt.erg/unyt.s)
    return ordered_sampled_Lbol


def assign_sampled_luminosities(particle_data, ordered_sampled_Lbol, number_of_BH, sigma_Y, rng, quiet=False):
    """
    Assign the sampled luminosities to the haloes in rank order, the brightest to the highest ranked halo.
    
    If sigma_Y < 0, haloes are ranked by the accretion rate of their most massive black hole, then black holes that
    are not accreting (in random order), then haloes without a black hole (in random order). 
    Otherwise they are ranked by maximum circular velocity with sigma_Y dex of log-normal scatter.

    Returns the luminosity of each halo as a unyt.unyt_array [erg/s], in the order of particle_data.

    :param  particle_data:          halo data, see read_particle_data
    :type   particle_data:          dict
    :param  ordered_sampled_Lbol:   sampled luminosities, brightest first, see sample_luminosities
    :type   ordered_sampled_Lbol:   unyt.unyt_array
    :param  number_of_BH:           number of haloes
    :type   number_of_BH:           int
    :param  sigma_Y:                scatter added to the maximum circular velocity ranking [dex], or < 0 to rank by accretion rate
    :type   sigma_Y:                float
    :param  rng:                    random number generator of the ranks
    :type   rng:                    np.random.Generator
    :param  quiet:                  If True, do not print progress
    :type   quiet:                  boolean
    """
    output_Lbol = unyt.unyt_array(np.zeros(number_of_BH, dtype=np.float64), units=unyt.erg/unyt.s)

    if sigma_Y < 0:
        report("\nRanking via MMBH_Maccr (sigma<0)", quiet)
        ranking_value = np.zeros(number_of_BH)

        # accreting black holes: ranked by accretion rate, above 1
        accretion_rate = particle_data['BoundSubhalo/MostMassiveBlackHoleAccretionRate']
        ranking_value[accretion_rate > 0] += accretion_rate[accretion_rate > 0] + 1

        # black holes that are not accreting: between 0 and -1
        has_BH = particle_data['BoundSubhalo/MostMassiveBlackHoleID'] > 0
        not_accreting = (accretion_rate <= 0) & has_BH
        ranking_value[not_accreting] -= rng.random(np.sum(not_accreting))

        # haloes without a black hole: between -1 and -2
        has_no_BH = ~has_BH
        ranking_value[has_no_BH] = -1 - rng.random(np.sum(has_no_BH))
    else:
        report(f"\nRanking via Vmax & sigma (sigma={sigma_Y})", quiet)
        Vmax = particle_data['BoundSubhalo/MaximumCircularVelocity']
        ranking_value = Vmax * 10**(rng.normal(loc=0, scale=sigma_Y, size=len(Vmax))) / 10**(np.log(10)/2 * sigma_Y**2)

    output_Lbol[np.argsort(ranking_value)[::-1]] = ordered_sampled_Lbol[0:number_of_BH]
    report(f"Range of values assigned: ({output_Lbol.min():.3e}, {output_Lbol.max():.3e}) [{np.sum(output_Lbol):.3e}]", quiet)
    return output_Lbol


def xray_map_names_AGN(observation_band, observation_type):
    """
    Returns the name of an AGN map dataset, the lightcone_io X-ray map name followed by 'AGN',
    e.g. 'XrayROSATConvolvedPhotonsAGN'. Observation type 'Fx' is the energy flux map, e.g. 'XrayROSATEnergyFluxAGN'.

    :param  observation_band:   X-ray band, 'ROSAT', 'erosita-high' or 'erosita-low' (any case)
    :type   observation_band:   str
    :param  observation_type:   'photons_intrinsic', 'photons_convolved', 'energies_intrinsic', 'energies_convolved' or 'Fx'
    :type   observation_type:   str
    """
    band_names = {
        "rosat": "ROSAT", 
        "erosita-high": "ErositaHigh",
        "erosita-low": "ErositaLow"
        }
    type_names = {
        "photons_intrinsic": "IntrinsicPhotons", 
        "photons_convolved": "ConvolvedPhotons",
        "energies_intrinsic": "IntrinsicEnergies", 
        "energies_convolved": "ConvolvedEnergies",
        "Fx": "EnergyFlux"
        }
    return "Xray" + band_names[observation_band.lower()] + type_names[observation_type] + "AGN"


def add_map(outfile, name, map_data, quiet=False):
    """
    write a map as a dataset into an open file, writing it if it does not exist yet.
    Write the dataset as float64 with gzip (level 9) compression and shuffle.

    :param  outfile:    file of the shell's maps
    :type   outfile:    h5py.File
    :param  name:       map dataset name, see xray_map_names_AGN
    :type   name:       str
    :param  map_data:   map values to add
    :type   map_data:   np.ndarray
    :param  quiet:      If True, do not print the map sums
    :type   quiet:      boolean
    """
    if name in outfile:
        existing_map = outfile[name][:]
        report(f"\t{name}: sum over existing map pixels         = {np.sum(existing_map)}", quiet)
        map_data = existing_map + map_data
        del outfile[name]
    report(f"\t{name}: sum over existing and new map pixels = {np.sum(map_data)}", quiet)

    outfile.create_dataset(
        name,
        data=map_data,
        dtype=np.float64,
        compression="gzip",
        compression_opts=9,
        shuffle=True
    )


def write_to_map(args, map_shell_nr, map_data_count_rate, map_data_Fx):
    """
    Add a shell's AGN count rate and energy flux maps to its output file.
    
    Make the file and directory if they does not exist yet.

    :param  args:                   command line arguments, see the end of this file
    :type   args:                   argparse.Namespace
    :param  map_shell_nr:           lightcone shell number
    :type   map_shell_nr:           int
    :param  map_data_count_rate:    count rate map [photons/s, or photons/s/cm^2 for intrinsic photons]
    :type   map_data_count_rate:    np.ndarray
    :param  map_data_Fx:            energy flux map [erg/s/cm^2]
    :type   map_data_Fx:            np.ndarray
    """
    os.makedirs(args.output_base_dir, exist_ok=True)
    output_shell_filename = args.output_base_dir+f'/AGN_{args.nside}.shell_{map_shell_nr}.hdf5'
    if os.path.isfile(output_shell_filename):
        report(f"\nAdding AGN to existing shell\t{map_shell_nr:03d}\n", args.quiet)
    else:
        report(f"\nWriting new AGN shell\t{map_shell_nr:03d}\n", args.quiet)

    with h5py.File(output_shell_filename, "a") as outfile:
        add_map(outfile, xray_map_names_AGN(args.observation_band, 'Fx'), map_data_Fx, args.quiet)
        add_map(outfile, xray_map_names_AGN(args.observation_band, args.observation_type), map_data_count_rate, args.quiet)

    print(f"Wrote map to {output_shell_filename}", flush=True)


def check_observation_band(observation_band):
    """
    Raise a ValueError unless the AGN X-ray luminosity can be found in the band. 
    Only the ROSAT band (0.5-2 keV) and the Shen+2020 soft band is supported for now.

    :param  observation_band:   X-ray band of the map
    :type   observation_band:   str
    """
    if observation_band.lower() not in BOLOMETRIC_CORRECTION_BANDS:
        raise ValueError(f"AGN maps can only be made in the {list(BOLOMETRIC_CORRECTION_BANDS)} band(s) for now: the Shen+2020 bolometric correction used is for 0.5-2 keV, not the {observation_band} band")


def read_photon_index(flux_table_filename, observation_band, observation_type):
    """
    Returns the photon index of the power law a flux table is normalised by, 'NormPhotonIndex' attribute of its
    fBB dataset, as a float. 
    Raise a ValueError if it is missing.

    :param  flux_table_filename:    path to the broadband flux table
    :type   flux_table_filename:    str
    :param  observation_band:       X-ray band of the table, e.g. 'ROSAT'
    :type   observation_band:       str
    :param  observation_type:       observation type of the table, e.g. 'photons_convolved'
    :type   observation_type:       str
    """
    name = f"{observation_band}/{observation_type}/fBB"
    with h5py.File(flux_table_filename, "r") as infile:
        if name not in infile:
            raise ValueError(f"{flux_table_filename} has no {name} table")
        if "NormPhotonIndex" not in infile[name].attrs:
            raise ValueError(f"{name} in {flux_table_filename} has no NormPhotonIndex, rebuild it with build_flamingo_broadband_flux_table")
        photon_index = float(infile[name].attrs["NormPhotonIndex"])
    if photon_index == -1:
        raise ValueError(f"{name} in {flux_table_filename} was normalised by a given f_norm_flux, so its photon index "
                         "(NormPhotonIndex = -1) is not known for the K-correction")
    return photon_index


def luminosity_distance(coordinates, scalefactor):
    """
    Returns the luminosity distance D_L = comoving distance / scale factor, 
    in the units of the coordinates squared.

    :param  coordinates:    comoving coordinates of each particle relative to the observer
    :type   coordinates:    unyt.unyt_array
    :param  scalefactor:    scale factor of each particle
    :type   scalefactor:    np.ndarray
    """
    comoving_distance = np.sqrt(coordinates[:, 0]**2 + coordinates[:, 1]**2 + coordinates[:, 2]**2)
    return 4 * np.pi * (comoving_distance**2) * ((1/scalefactor)**2)


def AbundanceMatched_BH_to_AGN_Fx(particle_abundance_matched_luminosities,
                                    particle_scale_factor,
                                    particle_coordinates,
                                    particle_log_nH,
                                    zmin, zmax, observation_band, photon_index):
    """
    X-ray energy flux of each AGN in the observed band: the bolometric luminosity times the Shen+2020 bolometric
    correction gives the rest-frame band luminosity Lx, Lx / 4 pi D_L^2 is the flux in the observed band
    [E_min, E_max]/(1+z), and the K-correction (1+z)^(2-Gamma) gives the flux in the observed band [E_min, E_max] of the
    power law the flux table is normalised by.

    Returns a tuple: 
        (energy flux of each particle as a unyt.unyt_array [erg/s/cm^2], mask of the particles in the shell with a luminosity and a column density)

    :param  particle_abundance_matched_luminosities:    bolometric luminosity of each particle
    :type   particle_abundance_matched_luminosities:    unyt.unyt_array
    :param  particle_scale_factor:                      scale factor of each particle
    :type   particle_scale_factor:                      np.ndarray
    :param  particle_coordinates:                       comoving coordinates of each particle relative to the observer
    :type   particle_coordinates:                       unyt.unyt_array
    :param  particle_log_nH:                            log10 column density of each particle [cm^-2], 0 if none
    :type   particle_log_nH:                            np.ndarray
    :param  zmin:                                       minimum redshift of the shell
    :type   zmin:                                       float
    :param  zmax:                                       maximum redshift of the shell
    :type   zmax:                                       float
    :param  observation_band:                           X-ray band of the map, only 'ROSAT' for now
    :type   observation_band:                           str
    :param  photon_index:                               photon index of the power law the flux table is normalised by,
                                                        see read_photon_index
    :type   photon_index:                               float
    """
    check_observation_band(observation_band)
    BH_redshift = 1.0/particle_scale_factor - 1.0

    z_mask = (BH_redshift >= zmin) & (BH_redshift < zmax)
    zeros_mask = (particle_abundance_matched_luminosities.to_value(unyt.erg/unyt.s) > 0) & (particle_log_nH > 0)
    combined_mask = z_mask & zeros_mask

    Fx = unyt.unyt_array(np.zeros(len(particle_scale_factor), dtype=np.float64), units='erg/s/cm**2')
    if np.sum(combined_mask) == 0:
        return Fx, combined_mask

    # rest-frame band luminosity from the bolometric luminosity
    masked_luminosities = particle_abundance_matched_luminosities[combined_mask].to(unyt.erg/unyt.s)
    correction_factor = Shen_2020_bolometric_correation(masked_luminosities, BOLOMETRIC_CORRECTION_BANDS[observation_band.lower()])
    Lx_band = masked_luminosities/correction_factor

    # flux in the observed band [E_min, E_max]/(1+z)
    Ld = luminosity_distance(particle_coordinates[combined_mask].to(unyt.Mpc), particle_scale_factor[combined_mask])
    Fx[combined_mask] = Lx_band/Ld

    # bolometric correction to the observed band [E_min, E_max], which the flux table is normalised to
    Fx[combined_mask] *= (1. + BH_redshift[combined_mask])**(2. - photon_index)

    return Fx, combined_mask


def get_Xray_values_for_map(observation_band, observation_type,
                                flux_table,
                                particle_abundance_matched_luminosities,
                                particle_scale_factor,
                                particle_coordinates,
                                particle_log_nH,
                                zmin, zmax, photon_index):
    """
    Energy flux and count rate of each AGN particle. 
    The count rate is the energy flux times the normalised broadband flux of the flux table at the AGN's redshift and column density.

    Returns a tuple 
        (energy flux [erg/s/cm^2], count rate [photons/s, or photons/s/cm^2 for intrinsic photons], mask of the particles added to the map).

    :param  observation_band:                           X-ray band of the map, only 'ROSAT' for now
    :type   observation_band:                           str
    :param  observation_type:                           'photons_intrinsic' or 'photons_convolved'
    :type   observation_type:                           str
    :param  flux_table:                                 broadband flux table of the band and observation type
    :type   flux_table:                                 FluxTable
    :param  particle_abundance_matched_luminosities:    bolometric luminosity of each particle
    :type   particle_abundance_matched_luminosities:    unyt.unyt_array
    :param  particle_scale_factor:                      scale factor of each particle
    :type   particle_scale_factor:                      np.ndarray
    :param  particle_coordinates:                       comoving coordinates of each particle relative to the observer
    :type   particle_coordinates:                       unyt.unyt_array
    :param  particle_log_nH:                            log10 column density of each particle [cm^-2], 0 if none
    :type   particle_log_nH:                            np.ndarray
    :param  zmin:                                       minimum redshift of the shell
    :type   zmin:                                       float
    :param  zmax:                                       maximum redshift of the shell
    :type   zmax:                                       float
    :param  photon_index:                               photon index of the power law the flux table is normalised by
    :type   photon_index:                               float
    """
    if "convolved" in observation_type:
        count_rate = unyt.unyt_array(np.zeros(len(particle_scale_factor), dtype=np.float64), units=unyt.photon * unyt.s**-1)
    elif "intrinsic" in observation_type:
        count_rate = unyt.unyt_array(np.zeros(len(particle_scale_factor), dtype=np.float64), units=unyt.photon * unyt.s**-1 * unyt.cm**-2)
    else:
        raise ValueError("Did not recognise the observation type")

    Fx, combined_mask = AbundanceMatched_BH_to_AGN_Fx(
                                    particle_abundance_matched_luminosities,
                                    particle_scale_factor,
                                    particle_coordinates,
                                    particle_log_nH,
                                    zmin, zmax,
                                    observation_band,
                                    photon_index,
                                    )
    if np.sum(combined_mask) == 0:
        return Fx, count_rate, combined_mask

    int_fBBz = flux_table(
                1./particle_scale_factor[combined_mask]-1.,
                particle_log_nH[combined_mask],
                )
    count_rate[combined_mask] = Fx[combined_mask].to_value(unyt.erg * unyt.cm**-2 * unyt.s**-1) * int_fBBz

    return Fx, count_rate, combined_mask


def place_abundance_matched_agn(args, particle_data, zmin, zmax, theta, phi, point_source_tables, flux_table, photon_index):
    """
    Make AGN point source maps for a lightcone shell. 
    Sample a column density for each AGN, find its energy flux and count rate, rotate it on the sky 
    and add it to its HEALPix pixel. 
    Add AGN to the map in chuncks. The column densities are stored in particle_data['log_nH'].

    Returns a tuple 
        (energy flux map [erg/s/cm^2], count rate map [photons/s, or photons/s/cm^2 for intrinsic photons])

    :param  args:                   command line arguments, see the end of this file
    :type   args:                   argparse.Namespace
    :param  particle_data:          'Coordinates', 'ExpansionFactors' and 'AbundanceMatchedLuminosity' of the shell's
                                    haloes, see main
    :type   particle_data:          dict
    :param  zmin:                   minimum redshift of the shell
    :type   zmin:                   float
    :param  zmax:                   maximum redshift of the shell
    :type   zmax:                   float
    :param  theta:                  co-latitude rotation angle of the shell
    :type   theta:                  unyt.unyt_quantity
    :param  phi:                    longitude rotation angle of the shell
    :type   phi:                    unyt.unyt_quantity
    :param  point_source_tables:    tables sampling the column densities
    :type   point_source_tables:    PointSourceTables
    :param  flux_table:             broadband flux table of the band and observation type
    :type   flux_table:             FluxTable
    :param  photon_index:           photon index of the power law the flux table is normalised by
    :type   photon_index:           float
    """
    quiet = args.quiet
    nr_particles_tot = len(particle_data["ExpansionFactors"])
    report(f"Shell redshift range =\t({zmin}, {zmax})", quiet)
    report(f"BH particles redshift range =\t({1./particle_data['ExpansionFactors'].max()-1.}, {1./particle_data['ExpansionFactors'].min()-1.})", quiet)
    report(f"Rotation angles=\t{theta}, {phi}\n", quiet)

    npix = hp.nside2npix(args.nside)
    map_data_Fx = np.zeros(npix, dtype=np.float64)
    map_data_count_rate = np.zeros(npix, dtype=np.float64)
    if 'convolved' in args.observation_type:
        output_units_str = 's**-1'
    elif 'intrinsic' in args.observation_type:
        output_units_str = 's**-1 * cm**-2'

    nr_particles_read = 0
    nr_particles_added = 0
    nr_particles_in_mask = 0
    nr_particles_out_mask = 0
    n_pixels_updated = 0
    number_of_chunks = 1 + nr_particles_tot // CHUNK_SIZE

    rotate = (theta.value != 0.) or (phi.value != 0.)

    particle_data["log_nH"] = np.zeros(nr_particles_tot, dtype=np.float64)

    for chunk in range(number_of_chunks):
        start_id = chunk * CHUNK_SIZE
        end_id = min((chunk+1) * CHUNK_SIZE, nr_particles_tot)
        if start_id >= end_id:
            break

        # column densities of the particles with a luminosity
        chunk_Lbol = particle_data['AbundanceMatchedLuminosity'][start_id:end_id]
        has_Lbol = chunk_Lbol.to_value(unyt.erg/unyt.s) > 0
        if np.any(has_Lbol):
            chunk_redshift = 1./particle_data["ExpansionFactors"][start_id:end_id][has_Lbol] - 1.
            particle_data['log_nH'][start_id:end_id][has_Lbol] = np.log10(point_source_tables.sample_nH(chunk_Lbol[has_Lbol], chunk_redshift))

        chunk_coords = particle_data['Coordinates'][start_id:end_id, :]
        Fx, count_rate, combined_mask = get_Xray_values_for_map(
                                args.observation_band,
                                args.observation_type,
                                flux_table,
                                particle_abundance_matched_luminosities=chunk_Lbol,
                                particle_scale_factor=particle_data["ExpansionFactors"][start_id:end_id],
                                particle_coordinates=chunk_coords,
                                particle_log_nH=particle_data["log_nH"][start_id:end_id],
                                zmin=zmin,
                                zmax=zmax,
                                photon_index=photon_index,
                                )

        # combined_mask: True if particles are in the shell, with a luminosity and a column density. 
        nr_particles_in_mask += np.sum(combined_mask)
        nr_particles_out_mask += np.sum(~combined_mask)
        nr_particles_read += end_id - start_id
        if np.any(combined_mask):
            pos = chunk_coords[combined_mask]
            if rotate:
                # keep rotation quiet prints a message for each call
                with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
                    pos = rotate_particle_coordinates(pos, theta, phi)
            m_Fx = Fx[combined_mask]
            m_count_rate = count_rate[combined_mask]

            # vec2pix can't handle unyt arrays, and only the direction of the vector is needed
            if isinstance(pos, unyt.unyt_array):
                pos = pos.value
            pixel = hp.pixelfunc.vec2pix(args.nside, pos[:,0], pos[:,1], pos[:,2])

            np.add.at(map_data_Fx, pixel, m_Fx.to_value('erg * s**-1 * cm**-2'))
            np.add.at(map_data_count_rate, pixel, m_count_rate.to_value(output_units_str))
            n_pixels_updated += np.shape(pixel)[0]
            nr_particles_added += m_Fx.shape[0]

        percent_done = (nr_particles_read/nr_particles_tot)*100.0
        report("Processed %d of %d particles (%.2f%%)" % (nr_particles_read, nr_particles_tot, percent_done), quiet, end='\r')

    report("\tTotal number of particles added to map = %d" % nr_particles_added, quiet)
    report("\tTotal number of particles in mask      = %d" % nr_particles_in_mask, quiet)
    report("\tTotal number of particles outside mask = %d" % nr_particles_out_mask, quiet)
    report("\tTotal number of pixels updated         = %d" % n_pixels_updated, quiet)
    report(f"\tSum over Fx map pixels     =  {np.sum(map_data_Fx)}", quiet)
    report(f"\tSum over CR map pixels     =  {np.sum(map_data_count_rate)}", quiet)
    return map_data_Fx, map_data_count_rate


def make_abundance_matched_agn_map(args, shell_nr, particle_data, shell_box, point_source_tables, flux_table, photon_index):
    """
    Make AGN maps and write them to the output file.

    :param  args:                   command line arguments, see the end of this file
    :type   args:                   argparse.Namespace
    :param  shell_nr:               lightcone shell number
    :type   shell_nr:               int
    :param  particle_data:          'Coordinates', 'ExpansionFactors' and 'AbundanceMatchedLuminosity' of the shell's
                                    haloes, see main
    :type   particle_data:          dict
    :param  shell_box:              shell redshift file label, 'L1' or 'L2p8', see shell_redshift_box
    :type   shell_box:              str
    :param  point_source_tables:    tables sampling the column densities
    :type   point_source_tables:    PointSourceTables
    :param  flux_table:             broadband flux table of the band and observation type
    :type   flux_table:             FluxTable
    :param  photon_index:           photon index of the power law the flux table is normalised by
    :type   photon_index:           float
    """
    theta, phi = read_rotation_angles(args.map_rotation_filename)
    redshifts = np.loadtxt(flamingo_shell_redshift_file(shell_box), delimiter=",")
    zmin = redshifts[shell_nr,0]
    zmax = redshifts[shell_nr,1]

    map_data_Fx, map_data_count_rate = place_abundance_matched_agn(args, particle_data, zmin, zmax, theta[shell_nr], phi[shell_nr], point_source_tables, flux_table, photon_index)
    write_to_map(args, shell_nr, map_data_count_rate, map_data_Fx)


def main(args):
    """
    Abundance match the haloes in the QLF models redshift range and add the assigned AGN to the maps of
    each lightcone shell, once for each level of additional scatter given by sigma_Y.

    :param  args:   command line arguments, see the end of this file
    :type   args:   argparse.Namespace
    """

    # fail before reading any data if the band's X-ray luminosity is not known, or the flux table has no photon index
    check_observation_band(args.observation_band)
    photon_index = read_photon_index(args.xray_table_filename, args.observation_band, args.observation_type)
    report(f"K-correction with the flux table's photon index: {photon_index}", args.quiet)

    # random seed of the abundance matching ranks and the sampled column densities
    seed = 44
    rng = np.random.default_rng(seed)

    # collect AGN (Black Hole) particle data. 
    # e.g. read halo lightcones and corresponding properties from SOAP
    particle_data, qlf_redshift_range, qlf_redshift, cosmo, shell_box = get_particle_data_for_abundance_matching(args)

    map_shell_numbers = np.unique(particle_data['Lightcone/ShellNumber'])
    report(f"\nAll Shells to update:\t {map_shell_numbers} \n", args.quiet)
    if not hasattr(particle_data["Lightcone/HaloCentre"], "units"):
        particle_data["Lightcone/HaloCentre"] = unyt.unyt_array(particle_data["Lightcone/HaloCentre"], units=unyt.Mpc)

    numb_bh = len(particle_data['Lightcone/SnapshotNumber'])
    ordered_sampled_Lbol = sample_luminosities(args, numb_bh, qlf_redshift_range, qlf_redshift, cosmo)

    # column density and flux tables, loaded once for all shells
    if args.nH_table_filename == 'new':
        report("\nbuilding nH probability function tables...", args.quiet)
        nH_table = PointSourceTables.make_nH_probability_function_table(log_Lx_min=30., log_Lx_max=60., zmin=qlf_redshift_range[0], zmax=qlf_redshift_range[1]+0.01, dnH=0.1, dLx=0.01, dz=0.01)
    else:
        report("\nreading nH probability function tables...", args.quiet)
        nH_table = args.nH_table_filename
    point_source_tables = PointSourceTables(nH_table=nH_table, flux_table=args.xray_table_filename, seed=seed, out_of_range="clamp")
    flux_table = point_source_tables.get_flux_table(args.observation_band, args.observation_type)

    output_base_dir = args.output_base_dir
    for realisation_idx, scatter_val in enumerate(args.sigma_Y):
        args.output_base_dir = output_base_dir+f"_{scatter_val}dex"
        report(f"\n+SCATTER: {scatter_val}\t[realisation numb {realisation_idx}]", args.quiet)
        report(f"\tAdditional log-normal scatter: {scatter_val:.2f} [dex]", args.quiet)
        report(f"base directory: {args.output_base_dir}\n", args.quiet)

        particle_data['AbundanceMatchedLuminosity'] = assign_sampled_luminosities(particle_data, ordered_sampled_Lbol, numb_bh, scatter_val, rng, quiet=args.quiet)

        for shell_nr in map_shell_numbers:
            report(f"\nComputing Xray values and creating Map for shell: {shell_nr}", args.quiet)
            in_shell = particle_data["Lightcone/ShellNumber"] == shell_nr
            if np.sum(in_shell) == 0:
                report(f"\n\nshell {shell_nr} has no particles assigned :(\n\n", args.quiet)
                continue
            shell_particle_data = {
                "Coordinates": particle_data["Lightcone/HaloCentre"][in_shell],
                "ExpansionFactors": 1./(1. + particle_data['Lightcone/Redshift'][in_shell]),
                "AbundanceMatchedLuminosity": particle_data["AbundanceMatchedLuminosity"][in_shell],
            }
            make_abundance_matched_agn_map(args, shell_nr, shell_particle_data, shell_box, point_source_tables, flux_table, photon_index)

        del particle_data['AbundanceMatchedLuminosity']

    args.output_base_dir = output_base_dir


if __name__=="__main__":

    argparser = argparse.ArgumentParser(description="AGN X-ray maps of the FLAMINGO lightcone shells from abundance matching the Shen+2020 QLF")

    # simulation
    argparser.add_argument("--box_res", type=str, required=True, help="boxsize and resolution, e.g. L1000N1800")
    argparser.add_argument("--sim_name", type=str, required=True, help="simulation name")
    argparser.add_argument("--lightcone_nr", type=int, default=0, help="lightcone number")

    # maps
    argparser.add_argument("--nside", type=int, required=True, help="nside of the output map")
    argparser.add_argument("--observation_type", type=str, required=True, help="'photons_intrinsic', 'photons_convolved'")
    argparser.add_argument("--observation_band", type=str, default="ROSAT", help="FLAMINGO X-ray band name, only 'ROSAT' for now")
    argparser.add_argument("--xray_table_filename", type=str, required=True, help="path/to/file/of/broadband/flux/table.hdf5")
    argparser.add_argument("--nH_table_filename", type=str, default="new", help="path/to/file/of/nH/probability/table.hdf5, or 'new' to build one")
    argparser.add_argument("--map_rotation_filename", type=str, required=True, help="path/to/file/with/map/shell/rotations.h5")

    # output
    argparser.add_argument("--output_base_dir", type=str, required=True, help="output directory, '_<scatter>dex' is appended for each scatter")
    argparser.add_argument("--quiet", action="store_true", help="only print one line for each map written")

    # abundance matching
    argparser.add_argument("--qlf_shell_nr", type=int, required=True, help="row of the QLF parameter file (tabulated QLF redshift) to abundance match")
    argparser.add_argument("--qlf_filename", type=str, required=True, help="path/to/qlf/data/filename")
    argparser.add_argument("--Log_Lmin", type=float, default=30, help="enforce a minimum sampled bolometric luminosity")
    argparser.add_argument("--sigma_Y", type=float, nargs='+', required=True, help="scatter in dex added to abundance matching order")

    args = argparser.parse_args()
    args.sigma_Y = np.asarray(args.sigma_Y).astype(float)

    main(args)

