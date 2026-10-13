#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J AGN_QLF_plots
#SBATCH -o ./logs/%x.%j.out
#SBATCH -e ./logs/%x.%j.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 12:00:00
#
# Plot the QLF of the abundance matched black holes against the Shen+2020 model, for a few rows of the QLF parameter
# file (tabulated QLF redshifts). The luminosities are sampled and assigned exactly as in make_agn_maps.sh (same
# functions and seed), so the plots check the AGN that go on the maps.
#
# Writes, for each row, <plot_dir>/abundance_matched_qlf_row<row>.png and .txt (the binned QLF and the model), and
# with --per_shell the QLF of each lightcone shell of the row to ..._shells.png.
#
# Edit the paths below, then:
#   mkdir -p logs; sbatch examples/plot_abundance_matched_qlf.sh
#

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

export OMP_NUM_THREADS=30


# python environment with lightcone_io, swiftlet and virgodc, and HEASoft for PyXspec
export HEADAS=/cosma/local/HEASoft/heasoft-6.36/x86_64-pc-linux-gnu-libc2.34/ # cosma install of HEAsoft
source /cosma/apps/do012/dc-mcdo1/swiftlet_env2/bin/activate

source ${HEADAS}/headas-init.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cosma/local/gcc/14.1.0/lib64 # gcc on cosma 

repo_dir=/cosma8/data/do012/dc-mcdo1/FLAMINGO_PointSource_Example
box_res=L1000N1800
sim_name=HYDRO_FIDUCIAL
lightcone_nr=0
qlf_filename=${repo_dir}/QLF_models/Shen2020_local_polished.dat
plot_dir=${repo_dir}/plots/${box_res}_${sim_name}_lightcone${lightcone_nr}
mkdir -p ${plot_dir}
qlf_rows="0 4 8"         # rows of the QLF file, as numbered by --qlf_shell_nr
sigma_Y=-1               # < 0: rank by accretion rate, >= 0: by Vmax with this scatter [dex]

cd ${repo_dir}

for qlf_row in ${qlf_rows}; do
    echo "QLF row ${qlf_row}"
    python -m flamingo_point_sources.plot_qlf \
        --box_res ${box_res} --sim_name ${sim_name} --lightcone_nr ${lightcone_nr} \
        --qlf_shell_nr ${qlf_row} --qlf_filename ${qlf_filename} \
        --sigma_Y ${sigma_Y} \
        --output ${plot_dir}/abundance_matched_qlf_row${qlf_row}.png \
        --per_shell || exit 1
done