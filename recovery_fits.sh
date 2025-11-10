#!/bin/bash -l

#SBATCH --job-name=fits
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=5-00:00:00
#SBATCH --partition compute
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=csahiti07@gmail.com

# define and create a unique scratch directory
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# copy the necessary files to scratch directory
cp ${SLURM_SUBMIT_DIR}/fit_params_mle.npy ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/recovery_fits.py ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/mle.py ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/likelihoods.py ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/constants.py ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/gen_data.py ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/task_structure.py ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/mdp_algms.py ${SCRATCH_DIRECTORY}

# activate Anaconda work environment
source /home/${USER}/.bashrc
conda activate env

# execute the job and time it
# time mpirun
python recovery_fits.py

# save the result to the submit directory
# cp -r ${SCRATCH_DIRECTORY}/result_recovery.npy ${SLURM_SUBMIT_DIR}/${SLURM_JOBID}

# step back and remove and scratch directory
cd ${SLURM_SUBMIT_DIR}
# rm -rf ${SCRATCH_DIRECTORY}

exit 0
