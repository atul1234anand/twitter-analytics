#!/bin/sh
#
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100000
#SBATCH --job-name="example-general-compute-job"
#SBATCH --output=create_dataset.out
#SBATCH --mail-user=atulanan@buffalo.edu
#SBATCH --mail-type=end
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cluster=ub-hpc
#SBATCH --constraint=CPU-Gold-6130
#Let's start some work

module load anaconda3 cuda/11.7.1 

eval "$(/cvmfs/soft.ccr.buffalo.edu/versions/2023.01/easybuild/software/Core/anaconda3/2022.05/bin/conda shell.bash hook)"

conda activate twitter_analytics

python3 /projects/academic/erdem/atulanan/twitter_analytics/Models/GCN/dataset.py
