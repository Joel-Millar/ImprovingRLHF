#!/bin/bash -l
#SBATCH -p gpulowbig
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -n 12
#SBATCH -J PEBBLE_quadruped_walk_2000_oracle
#SBATCH -o quad-%j.out

ulimit -s unlimited

#ADD virtual environment once established??

conda activate base

echo "This program is running on: $HOSTNAME"

# 0 - uniform : 6 - custom sampling : 7 - custom clustering

~/PEBBLE/BPref/scripts/quadruped_walk/1000/mistake/run_PEBBLE.sh 0

echo "This program has finished runnning on $HOSTNAME"
date
date_end = `date +%s`
seconds=$((date_end - date_start))
mins=$((seconds/60))
seconds=$((seconds-60*mins))
hours=$((mins/60))
mins=$((mins-60*hours))
echo Total run time: $hours Hours $mins Minutes $seconds Seconds
