#!/bin/bash
# instead of cgexec that is deprecated in rhel9, we simply bind this bash
# process to the cgroup, then nvidia-smi will be able to see the GPU
# of the cgroup of the running job
uid=$1
job=$2
echo $$ >> /sys/fs/cgroup/devices/slurm/uid_${uid}/job_${job}/tasks
nvidia-smi -L
