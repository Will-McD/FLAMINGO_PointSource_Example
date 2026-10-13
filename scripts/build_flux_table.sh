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
# observation types, with the basic template (absorbed power law with reflection, photon index 1.85) for 
# a given redshift range (redhsift_min - redshift_max) and column densities 10^20 - 10^26 [cm^-2]. 
#
# The table is built serially by xspec: 3 bands x 2 types x 300 redshifts x 61 column densities. 
# For a quick test use e.g. --dz 0.25 --dnH 1 --numb_energy_bins 1024.
#
# Edit the paths below, then: mkdir -p logs; sbatch examples/build_flux_table.sh
#


export OMP_NUM_THREADS=30

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4



# python environment with lightcone_io, swiftlet and virgodc, and HEASoft for PyXspec
export HEADAS=/cosma/local/HEASoft/heasoft-6.36/x86_64-pc-linux-gnu-libc2.34/ # cosma install of HEAsoft
source /cosma/apps/do012/dc-mcdo1/lightcone_env/bin/activate
source ${HEADAS}/headas-init.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cosma/local/gcc/14.1.0/lib64 # gcc on cosma 

repo_dir=/cosma8/data/do012/dc-mcdo1/FLAMINGO_PointSource_Example
snapshot=/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.0.hdf5

redshift_min=0
redshift_max=1
output=/snap8/scratch/dp004/dc-mcdo1/AGN_Xray_Table/AGN_flux_table_basic_z${redshift_min}-${redshift_max}.hdf5

cd ${repo_dir}

python -m flamingo_point_sources.build_flux_table \
    --snapshot ${snapshot} \
    --output ${output} \
    --observation_bands ROSAT eROSITA-low eROSITA-high \
    --observation_types photons_convolved photons_intrinsic \
    --zmin ${redshift_min} --zmax ${redshift_max} --dz 0.02 \
    --log_nH_min 20 --log_nH_max 26 --dnH 0.1 \
    --numb_energy_bins 16384 \
    --spec_index 1.85
