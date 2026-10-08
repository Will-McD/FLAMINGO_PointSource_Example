#!/bin/env python
"""
Build a broadband flux table of AGN spectra for the FLAMINGO X-ray bands, see PointSourceTables.build_flamingo_broadband_flux_table
and docs/PointSources.md. The spectra are the basic template (absorbed power law with reflection) with the photon
index given, normalised to the energy flux of their intrinsic power law.

Run from the top of the repository, with HEASoft (PyXspec) initialised, e.g.
    python -m flamingo_point_sources.build_flux_table --snapshot flamingo_0077.0.hdf5 --output AGN_flux_table.hdf5
"""
import argparse

from .agn_spectra import PointSourceTables, configure_xspec


def main(args):
    """
    Set up xspec from the snapshot and build the flux table.

    :param  args:   command line arguments, see the end of this file
    :type   args:   argparse.Namespace
    """
    configure_xspec(args.snapshot, abund=args.abund, xsect=args.xsect)
    path = PointSourceTables.build_flamingo_broadband_flux_table(
        args.observation_types, args.observation_bands, args.output, output_dir=args.output_dir,
        log_nH_min=args.log_nH_min, log_nH_max=args.log_nH_max, dnH=args.dnH,
        zmin=args.zmin, zmax=args.zmax, dz=args.dz,
        numb_energy_bins=args.numb_energy_bins, spec_index=args.spec_index,
        progress_bar=not args.quiet,
    )
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Build a broadband flux table of AGN spectra for the FLAMINGO X-ray bands")

    # xspec
    argparser.add_argument("--snapshot", type=str, required=True, help="FLAMINGO snapshot file (or directory) for the xspec cosmology")
    argparser.add_argument("--abund", type=str, default="aspl", help="xspec abundance table")
    argparser.add_argument("--xsect", type=str, default="vern", help="xspec photoionisation cross sections")

    # table
    argparser.add_argument("--output", type=str, required=True, help="output file name, '.hdf5' is appended if missing")
    argparser.add_argument("--output_dir", type=str, default=None, help="output directory, if not in the file name")
    argparser.add_argument("--observation_bands", type=str, nargs="+", default=["ROSAT", "eROSITA-low", "eROSITA-high"], help="X-ray bands")
    argparser.add_argument("--observation_types", type=str, nargs="+", default=["photons_convolved", "photons_intrinsic"], help="observation types")
    argparser.add_argument("--zmin", type=float, default=0., help="minimum redshift")
    argparser.add_argument("--zmax", type=float, default=6., help="maximum redshift")
    argparser.add_argument("--dz", type=float, default=0.02, help="width of the redshift bins")
    argparser.add_argument("--log_nH_min", type=float, default=20., help="minimum log10 column density [cm^-2]")
    argparser.add_argument("--log_nH_max", type=float, default=26., help="maximum log10 column density [cm^-2]")
    argparser.add_argument("--dnH", type=float, default=0.1, help="width of the log10 column density bins")
    argparser.add_argument("--numb_energy_bins", type=int, default=32768, help="photon energy intervals across each band, a power of two")
    argparser.add_argument("--spec_index", type=float, default=1.85, help="photon index of the spectra and of the normalisation")
    argparser.add_argument("--quiet", action="store_true", help="no progress bar")

    main(argparser.parse_args())
