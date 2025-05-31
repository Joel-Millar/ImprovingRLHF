#!/bin/bash -l
#SBATCH -p gpulowbig
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -n 12
#SBATCH -J PEBBLE_test
#SBATCH -t 12:00:00

ulimit -s unlimited

#ADD virtual environment once established??

date
echo "This program is running on: $HOSTNAME"

~/PEBBLE/BPref/scripts/walker_walk/1000/oracle/run_PEBBLE_test.sh 7

echo "This program has finished runnning on $HOSTNAME"
date
