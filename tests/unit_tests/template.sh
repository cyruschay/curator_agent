#!/bin/bash
#SBATCH --job-name=OLLAMA
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/slurm-%j_output.txt    # Standard output file (%j expands to jobID)
#SBATCH --error=slurm_logs/slurm-%j_error.txt      # Standard error file
#SBATCH --time=00:05:00           # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:2
#SBATCH --mem=12G                 # Memory per node

###################################################
# Module Loading
#module load python3/3.9.2
module load ollama
# Set environmental variables
export SOME_API_KEY="some_value"
# Set conda environment


###################################################
start_time=$SECONDS
echo "============== Slurm Job Info =============="
echo " Job ID:           $SLURM_JOB_ID"
echo " Job Name:         ${SLURM_JOB_NAME:-N/A}"
echo " User:             ${SLURM_JOB_USER:-$USER}"
echo " Account:          ${SLURM_JOB_ACCOUNT:-N/A}"
echo " Partition:        ${SLURM_JOB_PARTITION:-N/A}"
echo " Node List:        ${SLURM_JOB_NODELIST:-N/A}"
echo
echo " Submit Host:      ${SLURM_SUBMIT_HOST:-N/A}"
echo " Submit Directory: ${SLURM_SUBMIT_DIR:-N/A}"
echo " Work Directory:   $(pwd)"
echo
echo " Allocated CPUs:   $SLURM_CPUS_ON_NODE"
echo " Allocated GPUs:   $SLURM_JOB_GPUS"
echo
echo " Current Time:     $(date +%T)"
echo "============================================"


###################################################
ollama serve &
python test_graph.py


###################################################
echo "============================================"
end_time=$SECONDS
echo " Finish Time:      $(date +%T)"
echo " Time Taken:       $((end_time - start_time)) seconds"
