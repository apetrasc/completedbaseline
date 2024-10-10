#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A NAISS2024-1-13 -p alvis
# The name of the script is myjob
#SBATCH -J b24c01

# Time that will be given to this job
#SBATCH -t 00:29:00

# Number of nodes
#SBATCH -N 1 --gpus-per-node=A100:2
# Number of MPI processes per node (the following is actually the default)
# SBATCH --ntasks-per-node=4

#module load TensorFlow/2.5.0-fosscuda-2020b
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source sadanori/bin/activate
#SBATCH -e tmp/sb_error.e
#SBATCH -o tmp/sb_output.o

# Run the executable named myexe 
# and write the output into my_output_file
python3 CNN-predict_1.py > tmp/predict_1.txt
