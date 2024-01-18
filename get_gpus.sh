#!/bin/bash
# instead of cgexec that is deprecated in rhel9, we simply bind this bash
# process to the cgroup, then nvidia-smi will be able to see the GPU
# of the cgroup of the running job
task_file=$1
echo $$ >> "$task_file"
nvidia-smi -L
