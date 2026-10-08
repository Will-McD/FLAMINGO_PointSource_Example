#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH -J AGN_maps_ROSAT
#SBATCH -o ./logs/%x.%j.out
#SBATCH -e ./logs/%x.%j.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#
# Make ROSAT AGN maps of the lightcone shells for redshifts 0 - 6, by abundance matching each Shen+2020 QLF redshift
# (one row of the QLF parameter file) in turn. Each row adds its AGN to the shell maps, so the rows build up the full
# maps in the same output directories. See docs/AbundanceMatching.md.
#
# qlf_shells are the rows of the QLF file whose redshift range lies in 0 - z_max, and within the lightcone shells of
# the box: the FLAMINGO shell redshift files reach z = 3 for the L1000 boxes and z = 5 for L2800N5040.
#
# Needs the flux table from examples/build_flux_table.sh. Edit the paths below, then:
#   mkdir -p logs; sbatch examples/make_agn_maps.sh
#

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

# python environment with lightcone_io, swiftlet and virgodc, and HEASoft for PyXspec
source /cosma/apps/do012/dc-mcdo1/lightcone_env/bin/activate
source ${HEADAS}/headas-init.sh

repo_dir=/cosma8/data/do012/dc-mcdo1/Example_abundance_matching
box_res=L1000N1800
sim_name=HYDRO_FIDUCIAL
lightcone_nr=0
nside=4096
qlf_filename=${repo_dir}/QLF_models/Shen2020_local_polished.dat
flux_table=${repo_dir}/AGN_tables/AGN_flux_table_basic_z0-6.hdf5
rotations=/path/to/shell_rotations.hdf5            # 'shells/theta', 'shells/phi' [radians] for each shell
output_base_dir=/cosma8/data/do012/dc-mcdo1/AGN_maps/${box_res}_${sim_name}_lightcone${lightcone_nr}/ROSAT
sigma_Y="-1 0.3"                                    # < 0: rank by accretion rate, >= 0: by Vmax with this scatter [dex]
z_max=6

cd ${repo_dir}

# rows of the QLF file in 0 - z_max and inside the box's lightcone shells, as numbered by --qlf_shell_nr
qlf_shells=$(python - ${qlf_filename} ${box_res} ${z_max} <<'EOF'
import sys
import numpy as np
from swiftlet.swift_snapshot_redshift_conversion import flamingo_shell_redshift_file
qlf_filename, box_res, z_max = sys.argv[1], sys.argv[2], float(sys.argv[3])
z = np.genfromtxt(qlf_filename, names=True)["z"]
shell_box = "L1" if box_res.startswith("L1000") else "L2p8"
z_last_shell = np.loadtxt(flamingo_shell_redshift_file(shell_box), delimiter=",")[-1, 1]
rows = []
for i, zi in enumerate(z):
    # redshift range of the row, as in get_particle_data_for_abundance_matching
    z_hi = zi if i == len(z)-1 else zi + 0.5*(z[i+1] - zi)
    if z_hi <= min(z_max, z_last_shell) + 1e-6:
        rows.append(i)
print(" ".join(str(i) for i in rows))
EOF
)
echo "QLF rows to abundance match: ${qlf_shells}"

# the maps are added to, so start from empty output directories
if ls -d ${output_base_dir}_*dex >/dev/null 2>&1; then
    echo "Output directories ${output_base_dir}_*dex exist already, remove them or change output_base_dir" >&2
    exit 1
fi

for qlf_shell_nr in ${qlf_shells}; do
    echo "QLF row ${qlf_shell_nr}"
    python -m flamingo_point_sources.abundance_matched_map \
        --box_res ${box_res} --sim_name ${sim_name} --lightcone_nr ${lightcone_nr} \
        --nside ${nside} --observation_type photons_convolved --observation_band ROSAT \
        --xray_table_filename ${flux_table} \
        --map_rotation_filename ${rotations} \
        --output_base_dir ${output_base_dir} \
        --qlf_shell_nr ${qlf_shell_nr} --qlf_filename ${qlf_filename} \
        --sigma_Y ${sigma_Y} \
        --quiet || exit 1
done
