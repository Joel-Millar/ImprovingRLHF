#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -n 12
#SBATCH -J PEBBLE_walker_walk_1000_oracle
#SBATCH -o dis_walk-%j.out

ulimit -s unlimited

#ADD virtual environment once established??
conda activate base

date
echo "This program is running on: $HOSTNAME"

# Params: 0 - Uniform (defualt);  6 - custom_sampling; 7 - custom_clustering_sampling

~/PEBBLE/BPref/scripts/walker_walk/1000/oracle/run_PEBBLE.sh 6

echo "This program has finished runnning on $HOSTNAME"
date
