# Example abundance matching

This repository contains an except of code used to reproduce the FLAMINGO all-sky maps of AGN point-sources as presented in [McDonald et al (2026)](https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/stag1746/8796206?login=true). 
Note that the examples in this repository are specifically designed to run on COSMA and requires both [`HEASoft`](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/) and a `LightconeIO` environment (see [install instructions](https://lightconeio.readthedocs.io/en/latest/installation.html)). 

The resultant all-sky maps are available as part of the [FLAMINGO data release](https://dataweb.cosma.dur.ac.uk:8443/flamingo/index.html).


AGN point sources for the FLAMINGO lightcones: quasar bolometric luminosities are sampled from the Shen+2020 quasar
luminosity functions (QLFs), abundance matched to the haloes of the halo lightcone, given column densities and X-ray spectra,
and added to HEALPix X-ray maps of the lightcone shells.


## Documentation

- [AbundanceMatching](docs/AbundanceMatching.md): sampling the QLF with `SampleQLF`, and making AGN maps with
  `abundance_matched_map.py`
- [PointSources](docs/PointSources.md): AGN spectra with PyXspec (`AGNSpectra`), the column density table, and
  building and reading the broadband flux tables (`PointSourceTables`, `FluxTable`)



## AI use

Simple testing routines, code formatting, plotting scripts and the writing of documentation (`README.md`, [AbundanceMatching](docs/AbundanceMatching.md), [PointSources](docs/PointSources.md)) was done with the assistance of AI  (Claude Code, Opus 5.5 , Anthropic 2026). All use of AI was supervised, directed and vetted by the authors.


