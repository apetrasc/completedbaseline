#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A NAISS2023-22-1244 -p alvis

# The name of the script is myjob
#SBATCH -J b01c01

# Time that will be given to this job
#SBATCH -t 00:45:00

# Number of nodes
#SBATCH -N 1 --gpus-per-node=V100:2
# Number of MPI processes per node (the following is actually the default)
# SBATCH --ntasks-per-node=4

module load TensorFlow/2.5.0-fosscuda-2020b
module load matplotlib/3.5.2-foss-2022a

#SBATCH -e sb_error.e
#SBATCH -o sb_output.o

# Run the executable named myexe 
# and write the output into my_output_file
python3 classify.py
