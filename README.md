# Slurm-job-exporter
Prometheus exporter for the stats in the cgroup accounting with slurm. This will also collect stats of a job using NVIDIA GPUs.

## Requirements
Slurm need to be configured with `JobAcctGatherType=jobacct_gather/cgroup`. Stats are collected from the cgroups created by Slurm for each job. 

Python 3 with the following modules:

* `prometheus_client`
* `nvidia-ml-py` (optional)

If DCGM is installed and running, it will be used instead of NVML. DCGM have more GPU stats compared to NVML.

## Usage
```
usage: slurm-job-exporter.py [-h] [--port PORT]

Promtheus exporter for jobs running with Slurm within a cgroup

optional arguments:
  -h, --help   show this help message and exit
  --port PORT  Collector http port, default is 9798
``` 

## Sample
```
# HELP slurm_job_memory_usage Memory used by a job
# TYPE slurm_job_memory_usage gauge
slurm_job_memory_usage{account="group1",slurmjobid="1",user="user1"} 1.634453504e+010
slurm_job_memory_usage{account="group2",slurmjobid="2",user="user2"} 8.271761408e+09
# HELP slurm_job_memory_max Maximum memory used by a job
# TYPE slurm_job_memory_max gauge
slurm_job_memory_max{account="group1",slurmjobid="1",user="user1"} 1.6777220096e+010
slurm_job_memory_max{account="group2",slurmjobid="2",user="user2"} 1.9686723584e+010
# HELP slurm_job_memory_limit Memory limit of a job
# TYPE slurm_job_memory_limit gauge
slurm_job_memory_limit{account="group1",slurmjobid="1",user="user1"} 1.6777216e+010
slurm_job_memory_limit{account="group2",slurmjobid="2",user="user2"} 5.1539607552e+010
# HELP slurm_job_memory_cache bytes of page cache memory
# TYPE slurm_job_memory_cache gauge
slurm_job_memory_cache{account="group1",slurmjobid="1",user="user1"} 1.0655563776e+010
slurm_job_memory_cache{account="group2",slurmjobid="2",user="user2"} 8.0965632e+07
# HELP slurm_job_memory_rss bytes of anonymous and swap cache memory (includes transparent hugepages).
# TYPE slurm_job_memory_rss gauge
slurm_job_memory_rss{account="group1",slurmjobid="1",user="user1"} 5.452480512e+09
slurm_job_memory_rss{account="group2",slurmjobid="2",user="user2"} 7.846940672e+09
# HELP slurm_job_memory_rss_huge bytes of anonymous transparent hugepages
# TYPE slurm_job_memory_rss_huge gauge
slurm_job_memory_rss_huge{account="group1",slurmjobid="1",user="user1"} 1.23731968e+09
slurm_job_memory_rss_huge{account="group2",slurmjobid="2",user="user2"} 5.771362304e+09
# HELP slurm_job_memory_mapped_file bytes of mapped file (includes tmpfs/shmem)
# TYPE slurm_job_memory_mapped_file gauge
slurm_job_memory_mapped_file{account="group1",slurmjobid="1",user="user1"} 2.4803328e+08
slurm_job_memory_mapped_file{account="group2",slurmjobid="2",user="user2"} 4.933632e+07
# HELP slurm_job_memory_active_file bytes of file-backed memory on active LRU list
# TYPE slurm_job_memory_active_file gauge
slurm_job_memory_active_file{account="group1",slurmjobid="1",user="user1"} 1.53145344e+08
slurm_job_memory_active_file{account="group2",slurmjobid="2",user="user2"} 405504.0
# HELP slurm_job_memory_inactive_file bytes of file-backed memory on inactive LRU list
# TYPE slurm_job_memory_inactive_file gauge
slurm_job_memory_inactive_file{account="group1",slurmjobid="1",user="user1"} 8.699912192e+09
slurm_job_memory_inactive_file{account="group2",slurmjobid="2",user="user2"} 4.4879872e+07
# HELP slurm_job_memory_unevictable bytes of memory that cannot be reclaimed (mlocked etc)
# TYPE slurm_job_memory_unevictable gauge
slurm_job_memory_unevictable{account="group1",slurmjobid="1",user="user1"} 0.0
slurm_job_memory_unevictable{account="group2",slurmjobid="2",user="user2"} 0.0
# HELP slurm_job_core_usage_total Cpu usage of cores allocated to a job
# TYPE slurm_job_core_usage_total counter
slurm_job_core_usage_total{account="group1",core="1",slurmjobid="1",user="user1"} 1.165134620225e+012
slurm_job_core_usage_total{account="group1",core="5",slurmjobid="1",user="user1"} 1.209891619592e+012
slurm_job_core_usage_total{account="group2",core="4",slurmjobid="2",user="user2"} 5.711518455e+012
# HELP slurm_job_process_count Number of processes in a job
# TYPE slurm_job_process_count gauge
slurm_job_process_count{account="group1",slurmjobid="1",user="user1"} 1.0
slurm_job_process_count{account="group2",slurmjobid="2",user="user2"} 10.0
# HELP slurm_job_threads_count Number of threads in a job
# TYPE slurm_job_threads_count gauge
slurm_job_threads_count{account="group1",slurmjobid="1",user="user1"} 9.0
slurm_job_threads_count{account="group2",slurmjobid="2",user="user2"} 58.0
# HELP slurm_job_memory_usage_gpu Memory used by a job on a GPU
# TYPE slurm_job_memory_usage_gpu gauge
slurm_job_memory_usage_gpu{account="group1",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 2.3420993536e+010
slurm_job_memory_usage_gpu{account="group2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 1.5548284928e+010
# HELP slurm_job_power_gpu Power used by a job on a GPU in mW
# TYPE slurm_job_power_gpu gauge
slurm_job_power_gpu{account="group1",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 236037.0
slurm_job_power_gpu{account="group2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 248523.0
# HELP slurm_job_utilization_gpu Percent of time over the past sample period during which one or more kernels was executing on the GPU.
# TYPE slurm_job_utilization_gpu gauge
slurm_job_utilization_gpu{account="group1",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 39.71356095628427
slurm_job_utilization_gpu{account="group2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 92.63429206543753
# HELP slurm_job_utilization_gpu_memory Percent of time over the past sample period during which global (device) memory was being read or written.
# TYPE slurm_job_utilization_gpu_memory gauge
slurm_job_utilization_gpu_memory{account="group1",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 26.367425793004173
slurm_job_utilization_gpu_memory{account="group2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 42.77153803389558
# HELP slurm_job_sm_occupancy_gpu The ratio of number of warps resident on an SM. (number of resident as a ratio of the theoretical maximum number of warps per elapsed cycle)
# TYPE slurm_job_sm_occupancy_gpu gauge
slurm_job_sm_occupancy_gpu{account="group1",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 19.923513946712788
slurm_job_sm_occupancy_gpu{account="group2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 67.78674679458085
# HELP slurm_job_tensor_gpu The ratio of cycles the tensor (HMMA) pipe is active (off the peak sustained elapsed cycles)
# TYPE slurm_job_tensor_gpu gauge
slurm_job_tensor_gpu{account="group1",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 3.6616663501738054
slurm_job_tensor_gpu{account="group2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.6733508372047148
# HELP slurm_job_fp64_gpu Ratio of cycles the fp64 pipe is active
# TYPE slurm_job_fp64_gpu gauge
slurm_job_fp64_gpu{account="group1",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 0.0
slurm_job_fp64_gpu{account="group2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.0
# HELP slurm_job_fp32_gpu Ratio of cycles the fp32 pipe is active
# TYPE slurm_job_fp32_gpu gauge
slurm_job_fp32_gpu{account="group1",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 10.947193642079405
slurm_job_fp32_gpu{account="group2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 35.73380486655263
# HELP slurm_job_fp16_gpu Ratio of cycles the fp16 pipe is active
# TYPE slurm_job_fp16_gpu gauge
slurm_job_fp16_gpu{account="group1",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 0.0
slurm_job_fp16_gpu{account="group2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.0
# HELP slurm_job_pcie_gpu_total PCIe tx/rx bytes
# TYPE slurm_job_pcie_gpu_total counter
slurm_job_pcie_gpu_total{account="group1",direction="TX",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 9.6079424e+07
slurm_job_pcie_gpu_total{account="group1",direction="RX",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 8.1315216e+08
slurm_job_pcie_gpu_total{account="group2",direction="TX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 1.0392774e+07
slurm_job_pcie_gpu_total{account="group2",direction="RX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 4.4930668e+07
# HELP slurm_job_nvlink_gpu_total Nvlink tx/rx bytes
# TYPE slurm_job_nvlink_gpu_total counter
slurm_job_nvlink_gpu_total{account="group1",direction="TX",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 0.0
slurm_job_nvlink_gpu_total{account="group1",direction="RX",gpu="1",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 0.0
slurm_job_nvlink_gpu_total{account="group2",direction="TX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.0
slurm_job_nvlink_gpu_total{account="group2",direction="RX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.0
```
