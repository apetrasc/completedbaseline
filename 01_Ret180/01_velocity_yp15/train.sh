#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A NAISS2024-1-13 -p alvis

# The name of the script is myjob
#SBATCH -J b01c01

# Time that will be given to this job
#SBATCH -t 48:29:00

# Number of nodes
#SBATCH -N 1 --gpus-per-node=A100:4
# Number of MPI processes per node (the following is actually the default)
# SBATCH --ntasks-per-node=4

module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
#module load GPyTorch/1.9.1-foss-2021a-CUDA-11.3.1
source sadanori/bin/activate
#SBATCH -e sb_error.e
#SBATCH -o sb_output.o

# Run the executable named myexe 
# and write the output into my_output_file
python3 train.py > train/train.txt