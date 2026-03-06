#!/bin/bash
#SBATCH --job-name=ollama_0
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm-%j_output.txt    # Standard output file (%j expands to jobID)
#SBATCH --error=logs/slurm-%j_error.txt      # Standard error file
#SBATCH --time=48:00:00           # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=100567M      # Memory per node

###################################################
# Module Loading
module load ollama/0.13.5

# Set environmental variables
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_CONTEXT_LENGTH=20000

# Set conda environment


###################################################
start_time=$SECONDS
echo "============== Slurm Job Info =============="
echo " Job ID:           $SLURM_JOB_ID"
echo " Job Name:         ${SLURM_JOB_NAME:-N/A}"
echo " User:             ${SLURM_JOB_USER:-$USER} (${SLURM_JOB_ACCOUNT:-N/A})"
echo " Partition:        ${SLURM_JOB_PARTITION:-N/A} (${SLURM_JOB_NODELIST:-N/A})"
echo
echo " Submit Host:      ${SLURM_SUBMIT_HOST:-N/A}"
echo " Work Directory:   $(pwd)"
echo
echo " Allocated CPUs:   $SLURM_CPUS_ON_NODE"
echo " Allocated GPUs:   $SLURM_JOB_GPUS"
echo
echo " Start Time:     $(date +%T)"
echo "============================================"


###################################################
ollama serve > ollama.log 2>&1 & disown

python -u src/agent/graph.py


###################################################
echo "============================================"
end_time=$SECONDS
echo " Finish Time:      $(date +%T)"
echo " Time Taken:       $((end_time - start_time)) seconds"