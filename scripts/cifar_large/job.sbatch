#!/bin/bash
#SBATCH --time=24:00:00                  # Job run time (hh:mm:ss)
#SBATCH --account=ggnds-ic          # Replace "account_name" with an account available to you
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=16             # Number of task (cores/ppn) per node
#SBATCH --job-name=mnist_cert      # Name of batch job
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --output=cifar_cert.o%j        # Name of batch job output file
#SBATCH --gres=gpu:A100:1

# module load anaconda/2023-Mar/3

# bash linf.sh
# bash l2.sh
# bash joint.sh
# bash max.sh
# bash random.sh
# bash ours.sh