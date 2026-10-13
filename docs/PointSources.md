# AGN point sources: spectra and tables

This markdown document covers computing X-ray spectra of AGN with PyXspec, and the use of both column density (nH) distribution and the broadband flux tables.

| File | What it does |
|---|---|
| `flamingo_point_sources/agn_spectra.py` | `AGNSpectra` and `Spectrum` (spectra), `PointSourceTables` (nH and flux tables), `FluxTable` (reading a flux table) |
| `flamingo_point_sources/arf_utils.py` | reading telescope ARFs (effective areas) |
| `flamingo_point_sources/telescope_utils.py` | preset telescopes and the FLAMINGO X-ray bands |
| `ARF/` | ARF files, e.g. the eROSITA survey ARF |

For sampling luminosities and making maps see [AbundanceMatching](AbundanceMatching.md).

## Requirements

- HEASoft with PyXspec (`import xspec`), initialised with `$HEADAS/headas-init.sh`
- numba, h5py, astropy, scipy, unyt, tqdm
- LightconeIO (`lightcone_io`) for units and the snapshot cosmology

## Setting up xspec

```python
from flamingo_point_sources.agn_spectra import configure_xspec

cosmo = configure_xspec("/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.0.hdf5",
                        abund="aspl", xsect="vern")
```

`configure_xspec` sets the xspec cosmology from the snapshot, the abundance table and the
photoionisation cross sections. 
This must be called before computing spectra, otherwise the xspec defaults are used. 
It returns the snapshot cosmology.

## Telescopes and bands

| Telescope | Effective area |
|---|---|
| `"none"` | none (intrinsic spectra) |
| `"ROSAT"` | tabulated ROSAT effective area |
| `"eROSITA"` | `ARF/survey_tm8_arf_filter_2023-01-17.fits` |
| other | an ARF file in `ARF/` (`ARF_filename`, `ARF_reader`), or an analytic effective area (`ARF_filename="analytic"`, `ARF_reader=function of E [keV] -> cm^2`) |

`read_ARF` reads `.fits` (SPECRESP), `.hdf5` (`Energy`, `Area`) and comma-separated `.txt` ARFs; give a
`reader_function` for anything else.

The FLAMINGO X-ray bands (`FLAMINGO_BANDS`) are `rosat` 0.5-2 keV, `erosita-low` 0.2-2.3 keV and `erosita-high`
2.3-8 keV.

## Spectra: `AGNSpectra` and `Spectrum`

```python
import unyt
from flamingo_point_sources.agn_spectra import AGNSpectra

spec = AGNSpectra("eROSITA", E_band_min=0.2*unyt.keV, E_band_max=2.3*unyt.keV)

# an AGN at z = 1 with nH = 10^22 cm^-2 (nH_factor is in 10^22 cm^-2)
model, params = AGNSpectra.AGN_template(1.0, 1.0, spec_index=1.85, f_scatter=0.02)
spectrum = spec.compute_spectra(model, params, nbins=1025)

energy_flux, photon_flux = spectrum.flux()                 # through the effective area
energy_flux, photon_flux = spectrum.flux(effective_area=False)
bright = spectrum.normalised_to((1e-13, "erg/s/cm**2"))    # a new Spectrum with that intrinsic energy flux
```

`compute_spectra` returns a `Spectrum`, which holds the photon energies (`photon_energies`), the spectrum before
(`intrinsic_values`, photons/s/cm^2/keV) and after (`values`) the effective area, and the model and parameters used.

| `Spectrum` method | Returns |
|---|---|
| `flux(min_energy=None, max_energy=None, effective_area=True)` | (energy flux, photon flux) over the band or the given range |
| `scaled(factor)` | a new spectrum times factor |
| `normalised_to(target_flux, ...)` | a new spectrum with the given flux, e.g. `(1, "erg/s/cm**2")` or `(5, "photon/s")` |

The model can be

1. an xspec model string with a parameter dictionary, keyed by component and parameter name
   (`{"zpowerlw": {"PhoIndex": 1.8}}`), by component with values in order (`{"zpowerlw": [1.8, 0.1, 1]}`), or by
   parameter number (`{1: 0.005}`);
2. a function of the observed photon energy [keV] and its parameters, e.g. `lambda E, p: (E*(1+p[0]))**-p[1]`;
3. a pyxspec model.

`compute_spectra` can also scale the spectrum, `target_flux=(1, "erg/s/cm**2")`, or scale it by
`target_flux / norm_flux` (`norm_flux` as `(value, units)`). 
When `reuse_model=True`, repeated calls with the same model string and energy grid keep the xspec model and only set the parameters that changed (faster than updating all parameters at once, e.g., the column density changes).

### Templates

All templates (hard coded spectra models) take `(redshift, nH_factor, ...)` and return `(xspec model string, parameters)`:

| Template | Model |
|---|---|
| `basic_template` | `tbabs(ztbabs((constant*zpowerlw) + (constant*pexmon)))`: absorbed power law and reflection |
| `AGN_template` | as `basic_template` plus a scattered fraction `f_scatter` of the power law without intrinsic absorption |
| `SoftExcess_AGN_template` | as `AGN_template` plus a soft excess (`zbremss`) |

Other options: `spec_index` (photon index, 1.85), `MW_nH_factor` (Galactic nH, 0.018 x 10^22 cm^-2), `rel_refl`,
`cos_inc`, `fold_energy`, `ratio_abund`, `Fe_abund`, `norm_fact`. 
The reflection and absorbers are at the AGN's (i.e. the emitters) redshift.

## Column densities: the nH table

AGN column densities are drawn from the Ueda et al. (2014) distribution, which depends on redshift and the hard (2-10 keV) X-ray luminosity.

```python
from flamingo_point_sources.agn_spectra import PointSourceTables

table = PointSourceTables.make_nH_probability_function_table(
            log_nH_min=20., log_nH_max=26., log_Lx_min=35., log_Lx_max=55.,
            zmin=0., zmax=6., dnH=0.1, dLx=0.01, dz=0.01)
path = PointSourceTables.write_nH_table(table, "nH_table")   # or output_filename= in the call above
table = PointSourceTables.read_nH_table(path)
```

The table holds the bin edges (`z_edges`, `log_Lx_edges`, `log_nH_edges`), the column density bin centres
(`log_nH_mids`) and the probabilities (`probability_function`, shape (redshift, nH, Lx)).

Sampling a column density for each particle:

```python
pst = PointSourceTables(nH_table="nH_table.hdf5", seed=44)
nH = pst.sample_nH(L_bol, z)                  # cm^-2, 0 where L_bol <= 0
nH = pst.sample_nH(L_bol, z, part_ids=ids)    # identify particles by ID instead
```

`L_bol` is converted to the hard band with the Shen+2020 correction. 
Each particle's random number comes from the seed and the particle itself (its luminosity and redshift, or its ID), so for a given seed and table a particle always gets the same column density, however the particles are ordered or split into chunks. 
The sampler is a parallel numba kernel.

## Broadband flux tables

A flux table gives, for each X-ray band and observation type, the broadband flux of the AGN spectrum on a grid of
redshift and column density, normalised to the energy flux of its intrinsic power law in the observed band. The
count rate of an AGN is then its observed-band energy flux times the table.

### Building

From the command line, `python -m flamingo_point_sources.build_flux_table --snapshot <snapshot> --output <file>`
builds a table with the basic template (options as below, `--help` for the list).
`scripts/build_flux_table.sh` is a job building the table for all three bands and both observation types for z = 0 - 6. 

In Python:

```python
configure_xspec(snapshot_filename)
path = PointSourceTables.build_flamingo_broadband_flux_table(
            ["photons_convolved", "photons_intrinsic"], ["ROSAT", "eROSITA-low", "eROSITA-high"],
            "AGN_flux_table.hdf5",
            log_nH_min=20., log_nH_max=26., dnH=0.1,
            zmin=0., zmax=6., dz=0.02,
            numb_energy_bins=32768, spec_index=1.85)
```

| Argument | Meaning |
|---|---|
| `observation_types` | `photons_convolved` (through the telescope's effective area) and/or `photons_intrinsic` |
| `observation_bands` | `ROSAT`, `eROSITA-low`, `eROSITA-high`; convolved ROSAT uses the ROSAT response, the eROSITA bands the eROSITA ARF |
| `log_nH_*`, `dnH`, `z*`, `dz` | the grid: column density bins centred on `log_nH_min ... log_nH_max`, redshift bins from `zmin` to `zmax` |
| `numb_energy_bins` | energy intervals across the band; a power of two uses Romberg integration |
| `spec_index` | photon index of the default spectrum and normalisation |
| `f_spectra_z` | spectrum, a function of (redshift, nH_factor) returning (model, parameters); default `basic_template` with `spec_index` |
| `f_norm_flux` | normalising flux, a function of (redshift, E_min, E_max) returning (value, units); default the analytic energy flux of a power law with `spec_index` in the observed band |
| `output_dir` | directory; default `AGN_tables/` next to the module unless the file name has a directory |

The redshift loop is outermost and the column density loop inside it, so one xspec model is reused per redshift and
only its absorption changes. 
Building a full table is dominated by xspec; reduce the grid or `numb_energy_bins` for
tests.

### Table file structure

```
Bins/
    z_edges, Redshift          redshift bin edges and centres
    log_nH_edges, log_nH       log10 column density bin edges and centres [cm^-2]
    observation_bands, observation_types
spectrum/                      attributes only
    @model                     xspec model string
    @<component>.<parameter>   each model parameter, e.g. @zpowerlw.PhoIndex = 1.85;
                               the redshifts and column density as their [min, max]
<band>/<observation type>/fBB  shape (column densities, redshifts)
    @units                     units of the photon flux (1/s convolved, 1/(cm**2*s) intrinsic)
    @normalisation             what the table is divided by
    @telescope                 telescope of the effective area
    @energy_band_keV           [E_min, E_max]
    @NormPhotonIndex           photon index of the normalising power law (spec_index), -1 if f_norm_flux was given
DateString
```

The table value is the spectrum's photon flux divided by the normalising energy flux in erg/s/cm^2. So a table value
times an AGN's energy flux [erg/s/cm^2] in the observed band gives its count rate, in photons/s for
`photons_convolved` or photons/s/cm^2 for `photons_intrinsic`.


### Reading the tables

```python
import numpy as np
from flamingo_point_sources.agn_spectra import FluxTable, PointSourceTables

table = FluxTable("AGN_flux_table.hdf5", "ROSAT", "photons_convolved", out_of_range="clamp")
fBB = table(z, log_nH)                                    # one value per particle

pst = PointSourceTables(nH_table="nH_table.hdf5", flux_table="AGN_flux_table.hdf5", seed=44, out_of_range="clamp")
nH = pst.sample_nH(L_bol, z)
fBB = pst.flux("ROSAT", "photons_convolved", z, np.log10(nH))
count_rate = Fx_observed_band * fBB
```

The nearest column density bin is used, and the redshift is interpolated (`interpolate=True`) or given by the nearest
bin. Redshifts outside the table's bin centres give NaN (`out_of_range="nan"`) or the value at the nearest centre
(`"clamp"`). 

`FluxTable` reads the table once and evaluates any number of particles in one pass. This can be called in chunks to limit the peak memory requirements/
`normalised_broadband_flux_from_table` is a one-off version that loads the table on each call (reduce overhead memory cost).


### Bolometric correction

The table is normalised to the power law's flux in the **observed** band, which is the frame of xspec's spectra. A
rest-frame band luminosity L_band (e.g. from a bolometric correction) gives

    L_band / 4 pi D_L^2 = flux in the observed band [E_min, E_max] / (1+z)

so multiply it by (1+z)^(2 - Gamma), with Gamma the fBB dataset's `NormPhotonIndex`, before multiplying by the table.
`abundance_matched_map.py` does this; see [AbundanceMatching](AbundanceMatching.md).
