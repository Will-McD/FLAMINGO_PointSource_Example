# Abundance matching with the Shen+2020 QLF

This markdown document covers sampling quasar bolometric luminosities from the Shen et al. (2020) quasar luminosity function
(QLF) and assigning them to the haloes of a FLAMINGO halo lightcone, ending in HEALPIX maps of the X-ray emission from AGN point sources.

| File | What it does |
|---|---|
| `flamingo_point_sources/sample_qlf.py` | `SampleQLF`: the QLF, and sampling luminosities from it; `Shen_2020_bolometric_correation` |
| `QLF_models/` | the Shen+2020 QLF parameter files |
| `flamingo_point_sources/abundance_matched_map.py` | the abundance matching and map making script |
| `flamingo_point_sources/plot_abundance_matched_qlf.py` | plots the QLF of the abundance matched black holes against the model |

The spectra and tables used to turn luminosities into X-ray fluxes are described in [PointSources](PointSources.md).

## The QLF

The Shen+2020 QLF is a double power law in bolometric luminosity,

    phi(L) = phi* / ((L/L*)^gamma1 + (L/L*)^gamma2)        [Mpc^-3 dex^-1]

with best fit parameters tabulated at the redshifts they were measured at. 
The corresponding parameter files:

| File | Redshifts | Columns |
|---|---|---|
| `QLF_models/Shen2020_local_polished.dat` | 0.2 - 7.0 (34 rows) | `z, gamma1, gamma2, log_phi_star, log_L_star` (+ errors) |
| `QLF_models/Shen2020_local_free.dat` | 0.2 - 6.4 (31 rows) | `z, gamma1, gamma2, phi_s, L_s` (+ errors) |

`log_phi_star` [Mpc^-3 dex^-1] and `log_L_star` in Lsun. `phi_s` and `L_s` are renamed to `log_phi_star` and
`log_L_star` when read (`QLF_COLUMN_ALIASES`), and rows starting with `#` are skipped.


## `SampleQLF` 

```python
import unyt
from flamingo_point_sources.sample_qlf import SampleQLF

qlf = SampleQLF("Shen2020_local_polished.dat", seed=44)   # a path, or the name of a file in QLF_models/

# select the QLF of a shell: the tabulated row nearest its midplane redshift
qlf.update_qlf_redshift(1.03)
print(qlf.z_qlf, qlf.model_params)                         # 1.0, {'z': 1.0, 'gamma1': ..., ...}

# the QLF at some bolometric luminosities (erg/s if they have no units)
phi = qlf.qlf_values([1e44, 1e45, 1e46] * unyt.erg / unyt.s)   # Mpc^-3 dex^-1

# sample luminosities for 10^5 black holes in a shell of comoving volume 10^9 Mpc^3
L_bol, n_sampled = qlf.sample_LF_Lbol(1.03, 100000, 1e9 * unyt.Mpc**3)
```

`sample_LF_Lbol` selects the QLF and returns `(luminosities, number sampled)`. The luminosities [erg/s] are ordered from brightest to faintest. There are fewer sampled luminosities than black holes if the given lower limit is met before each black hole is sampled, i.e. if the volume does not hold `n_BHs` quasars brighter than `log_Lbol_min_value`.

### How the sampling works

1. The QLF is evaluated on a fine grid in log10 L (1/1000 of the bin width) and integrated from the bright end down,
   giving the number density brighter than each luminosity, N(>L).
2. Bins of `log_Lbol_bin_size` dex are stepped from `log_Lbol_max_value` down. The expected number of quasars in a
   bin is the difference of N(>L) at its edges times the volume, and the number sampled is a Poisson draw with that
   mean.
3. Each quasar is placed in its bin by inverse-CDF sampling of the QLF across the bin (`within_bin="qlf"`), so the
   luminosities follow the QLF's shape within the bin rather than being uniform in log L.
4. Sampling stops once `n_BHs + 1` luminosities are drawn (the last bin is cut) or `log_Lbol_min_value` is reached.


### Options

| Argument | Default | Meaning |
|---|---|---|
| `log_Lbol_bin_size` | 0.1 | bin width in log10 L |
| `log_Lbol_min_value`, `log_Lbol_max_value` | 30, 50 | log10 L range [erg/s] |
| `bright_end` | `"all"` | where sampling starts: `"all"` samples every bin, so rare bright quasars appear with their Poisson probability, whatever the bin width; `"cumulative"` starts where N(>L) V = `min_expected`; `"per_bin"` skips bins until one bin alone expects `min_expected` (the original method) |
| `min_expected` | 1 | used by `bright_end="cumulative"` and `"per_bin"` |
| `within_bin` | `"qlf"` | `"qlf"` (inverse CDF of the QLF in the bin) or `"uniform"` (uniform in log L) |
| `exact_bins` | True | bin edges `log_Lmax - k*dlogL` and expected numbers from N(>L); False uses Simpson's rule in each bin |
| `per_shell_seed` | True | seed each shell from `seed` and its midplane redshift (see below) |

The original method is `per_shell_seed=False, within_bin="uniform", exact_bins=False, bright_end="per_bin"`.

### Reproducibility

With `per_shell_seed=True`, each shell gets its own random number generator, seeded from `seed` and the exact value of
its midplane redshift (`shell_rng`). A shell's luminosities therefore do not depend on which shells were sampled
before it or in what order, so shells can be sampled independently, e.g. on different MPI ranks. Use the same
definition of the midplane redshift for every shell, since it chooses both the QLF row and the seed. With
`per_shell_seed=False` the instance's random stream continues from call to call.

### Bolometric corrections

`Shen_2020_bolometric_correation(Lbol, band)` returns L_bol / L_band for the Shen+2020 soft (0.5-2 keV) or hard
(2-10 keV) band:

    L_bol / L_band = c1 (L_bol / 10^10 Lsun)^k1 + c2 (L_bol / 10^10 Lsun)^k2

with the coefficients in `SHEN2020_COEFFS`, as given by 
([quasarlf](https://bitbucket.org/ShenXuejian/quasarlf)). 


## Checking the abundance matched QLF: `plot_qlf.py`

Plot the sampled QLF model with `plot_qlf.py`. 
For one QLF row, the script reads the haloes, samples and assigns the luminosities with the same functions and seed
as `abundance_matched_map.py`, and bins the assigned luminosities into a QLF,

    phi(L) = N(L) / (V dlogL)        [Mpc^-3 dex^-1]

with V the comoving volume the luminosities were sampled for. 

```bash
python -m flamingo_point_sources.plot_qlf \
    --box_res L1000N1800 --sim_name HYDRO_FIDUCIAL \
    --qlf_shell_nr 4 --qlf_filename QLF_models/Shen2020_local_polished.dat \
    --sigma_Y -1 --output plots/abundance_matched_qlf_row4.png --per_shell
```

| Argument | Required | Meaning |
|---|---|---|
| `--box_res`, `--sim_name`, `--lightcone_nr` | as in `abundance_matched_map.py` | simulation and lightcone |
| `--qlf_shell_nr`, `--qlf_filename`, `--Log_Lmin` | as in `abundance_matched_map.py` | QLF row and sampling |
| `--sigma_Y` | no (-1) | one scatter value [dex]; < 0 ranks by accretion rate |
| `--output` | yes | figure (.png); the binned QLF and the model are written next to it as `.txt` |
| `--dlogL` | no (0.25) | width of the plotted log10 luminosity bins |
| `--per_shell` | no | also plot the QLF of each lightcone shell in the row's range, to `<output>_shells.png` |
| `--quiet` | no | only print the files written |

What to expect:

- The QLF of the whole range does not depend on `sigma_Y`: the ranking only decides which halo gets which luminosity,
  and every ranking assigns the same sample. The per-shell QLFs do depend on it, since the ranking decides which
  shells' haloes get the bright luminosities, so `--per_shell` shows whether the ranking keeps the QLF's redshift
  dependence across the range.
- The plotted bins start at the faintest assigned luminosity, as the sampler stops a little above `Log_Lmin` (its
  last 0.1 dex bin is not drawn) and the bin it stops in is only partly filled.
- If there are fewer haloes than quasars brighter than `Log_Lmin` in the volume, every halo gets a luminosity and the
  faint end is cut where the haloes run out. This is marked on the plot ("ran out of haloes"); above it the QLF should
  follow the model.
- At the bright end the bins hold few quasars, so the ratio scatters with the Poisson errors shown.

See `scripts/plot_qlf.sh` for an example job plotting a few QLF rows.



## Making AGN point source maps: `abundance_matched_map.py`

For a QLF at a given redshift (one row of the QLF file), the script:

1. finds the redshift range of that row, the midpoint between the neighbouring QLF redshifts;
2. reads the haloes of the halo lightcone and corresponding SOAP catlaogues in that range, and assigns each halo to its lightcone shell;
3. samples one luminosity per halo from the QLF for the comoving volume;
4. assigns the sampled luminosities to the haloes in rank order, for each scatter value given:
   - `sigma_Y < 0`: by the accretion rate of the most massive black hole, then black holes that are not accreting
     (random order), then haloes without a black hole (random order);
   - `sigma_Y >= 0`: by Vmax with `sigma_Y` dex of log-normal scatter;
5. gives each AGN a column density from the Ueda+2014 distribution, before computing its X-ray energy flux and count rate. 
6. add each AGNsX-ray energy flux and count rate to the corresponding HEALPix map for the lightcones redshift shell.



### Important: SOAPIndex
The original version of the abundance matching code was built prior to the `sorted_hbt_halo_lightcones` and therefore mapped the halo lightcones to soap via the halo track ID (`InputHalos/HBTplus/DescendantTrackId`) instead of the lightcones `InputHalos/SOAPIndex` values. 
Therefore the errors with the miss-alignment of `InputHalos/SOAPIndex` did not affect published results. 
The example shown in `abundance_matched_map` uses the now corrected `InputHalos/SOAPIndex` values. 


### Running it

On one MPI rank, with HEASoft initialised, as the flux table code imports PyXspec:

```bash
python -m flamingo_point_sources.abundance_matched_map \
    --box_res L1000N1800 --sim_name HYDRO_FIDUCIAL \
    --nside 4096 --observation_type photons_convolved --observation_band ROSAT \
    --xray_table_filename AGN_flux_table.hdf5 \
    --map_rotation_filename shell_rotations.hdf5 \
    --output_base_dir AGN_maps \
    --qlf_shell_nr 4 --qlf_filename QLF_models/Shen2020_local_polished.dat \
    --sigma_Y -1 0.3
```

| Argument | Required | Meaning |
|---|---|---|
| `--box_res` | yes | box size and resolution, e.g. `L1000N1800` (`L1000N1800`, `L1000N3600` and `L2800N5040` have halo lightcones) |
| `--sim_name` | yes | simulation name, e.g. `HYDRO_FIDUCIAL` |
| `--lightcone_nr` | no (0) | lightcone number |
| `--nside` | yes | HEALPix nside of the maps |
| `--observation_type` | yes | `photons_convolved` or `photons_intrinsic` |
| `--observation_band` | no (`ROSAT`) | only `ROSAT` for now: the Shen+2020 soft correction is for 0.5-2 keV |
| `--xray_table_filename` | yes | broadband flux table, see [PointSources](PointSources.md); it needs `NormPhotonIndex` |
| `--nH_table_filename` | no (`new`) | column density table, or `new` to build one for the redshift range |
| `--map_rotation_filename` | yes | shell rotation angles (`shells/theta`, `shells/phi` in radians) |
| `--output_base_dir` | yes | output directory, `_<scatter>dex` is appended for each scatter |
| `--qlf_shell_nr` | yes | row of the QLF parameter file to abundance match |
| `--qlf_filename` | yes | QLF parameter file |
| `--Log_Lmin` | no (30) | minimum sampled log10 bolometric luminosity [erg/s], at least 30 |
| `--sigma_Y` | yes | one or more scatter values [dex], one set of maps each; < 0 ranks by accretion rate |
| `--quiet` | no | only print one line for each map written |

See `scripts/make_agn_maps.sh` for an example job running every QLF row in z = 0 - 6 in turn into the same output
directories.
The run stops before reading any data if the band is not ROSAT, or the flux table has no `NormPhotonIndex` (rebuild
it) or was normalised with a given `f_norm_flux` (`NormPhotonIndex = -1`).

### Output

One file per redshift shell for the ROSAT band:

| Dataset | Units |
|---|---|
| `XrayROSATEnergyFluxAGN` | [erg/s/cm^2], observed 0.5-2 keV energy flux |
| `XrayROSATConvolvedPhotonsAGN` | [photons/s] (`photons_convolved`) |
| `XrayROSATIntrinsicPhotonsAGN` | [photons/s/cm^2] (`photons_intrinsic`) |

If a shell's file exists already, the new values are added to its maps, so the QLF rows can be run one after another
into the same directories to build up the full maps.

### Notes

- A random seed of 44 is fixed in `main()` for the ranks and the column densities, and the QLF sample is seeded per
  shell, so a run is reproducible.
- The script runs on one MPI rank; `read_particle_data` takes an MPI communicator so it can be extended later.
- It needs `swiftlet` (ExtraSWIFTLightcones), LightconeIO (`lightcone_io`), `virgo` (virgodc), healpy and h5py. The
  FLAMINGO shell redshift files are found by swiftlet's `flamingo_shell_redshift_file`, and the `L1` or `L2p8` file
  is chosen from the box.

