# Example abundance matching

AGN point sources for the FLAMINGO lightcones: quasar bolometric luminosities are sampled from the Shen+2020 quasar
luminosity functions (QLFs), abundance matched to the haloes of the halo lightcone, given column densities and X-ray spectra,
and added to HEALPix X-ray maps of the lightcone shells.

## Documentation

- [AbundanceMatching](docs/AbundanceMatching.md): sampling the QLF with `SampleQLF`, and making AGN maps with
  `abundance_matched_map.py`
- [PointSources](docs/PointSources.md): AGN spectra with PyXspec (`AGNSpectra`), the column density table, and
  building and reading the broadband flux tables (`PointSourceTables`, `FluxTable`)

