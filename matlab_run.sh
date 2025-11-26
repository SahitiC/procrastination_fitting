#!/bin/bash -l

#SBATCH --output ./job.%j.out     # File containing output writen to stdout
#SBATCH --error  ./job.%j.err     # File containing output writen to stderr
#SBATCH -D ./                     # Initial working directory
#SBATCH --job-name cbm
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=5-00:00:00
#SBATCH --partition compute
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=csahiti07@gmail.com

# define and create a unique scratch directory
module purge                      # Purge the default modules loaded
module load singularity           # Load the singularity module

# Tell singularity which directories can be accessed from the container
export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"

# Setup any environment variables to be set inside the container SINGULARITYENV_<MY_VAR>
export SINGULARITYENV_MATLABPATH=$PWD

# Path to the Matlab container which sets the version.
export MATLAB_CONTAINER_PATH=/ptmp/containers/matlab-r2021a.sif

# Path to the Matlab container which sets the version.
export MATLAB_CONTAINER_PATH=/ptmp/containers/matlab-r2021a.sif

echo "Here we go..."
hostname
id
whoami
date --rfc-3339=ns
${MATLAB_CONTAINER_PATH} -nodesktop -nojvm -nosplash -r 'disp("Hello world!"); exit;'
date --rfc-3339=ns
echo " ... done."
