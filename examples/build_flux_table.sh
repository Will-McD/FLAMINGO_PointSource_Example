#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J AGN_flux_table
#SBATCH -o ./logs/%x.%j.out
#SBATCH -e ./logs/%x.%j.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#
# Build the AGN broadband flux table for the three FLAMINGO X-ray bands (ROSAT, eROSITA-low, eROSITA-high), both
# observation types, with the basic template (absorbed power law with reflection, photon index 1.85) for redshifts
# 0 - 6 and column densities 10^20 - 10^26 cm^-2. See docs/PointSources.md.
#
# The table is built serially by xspec: 3 bands x 2 types x 300 redshifts x 61 column densities. For a quick test
# use e.g. --dz 0.25 --dnH 1 --numb_energy_bins 1024.
#
# Edit the paths below, then: mkdir -p logs; sbatch examples/build_flux_table.sh
#

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

# python environment with lightcone_io, swiftlet and virgodc, and HEASoft for PyXspec
source /cosma/apps/do012/dc-mcdo1/lightcone_env/bin/activate
source ${HEADAS}/headas-init.sh

repo_dir=/cosma8/data/do012/dc-mcdo1/Example_abundance_matching
snapshot=/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.0.hdf5
output=${repo_dir}/AGN_tables/AGN_flux_table_basic_z0-6.hdf5

cd ${repo_dir}

python -m flamingo_point_sources.build_flux_table \
    --snapshot ${snapshot} \
    --output ${output} \
    --observation_bands ROSAT eROSITA-low eROSITA-high \
    --observation_types photons_convolved photons_intrinsic \
    --zmin 0 --zmax 6 --dz 0.02 \
    --log_nH_min 20 --log_nH_max 26 --dnH 0.1 \
    --numb_energy_bins 32768 \
    --spec_index 1.85
