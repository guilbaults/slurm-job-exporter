# Slurm-job-exporter

Prometheus exporter for the stats in the cgroup accounting with Slurm. This will also collect stats of a job using NVIDIA GPUs.

## Requirements

Slurm need to be configured with `JobAcctGatherType=jobacct_gather/cgroup`. Stats are collected from the cgroups created by Slurm for each job.

Python 3 with the following modules:

* `prometheus_client`
* `nvidia-ml-py` (optional)

DCGM is recommended when NVIDIA GPUs are installed:

* If DCGM is installed and running, it will be used instead of NVML.
* NVLINK and and a few other stats are only available with DCGM.
* MIG devices are supported.

`nvidia-smi -L` is run in each cgroup to detect which GPU is allocated to a Slurm job.

## Usage

```plaintext
usage: slurm-job-exporter.py [-h] [--port PORT] [--monitor MONITOR]
                             [--dcgm-update-interval DCGM_UPDATE_INTERVAL]

Promtheus exporter for jobs running with Slurm within a cgroup

optional arguments:
  -h, --help            show this help message and exit
  --port PORT           Collector http port, default is 9798
  --monitor MONITOR     GPU data monitor [dcgm|nvml], default is dcgm
  --dcgm-update-interval DCGM_UPDATE_INTERVAL
                        DCGM update interval in seconds, default is 10
```

## Cgroup v1 vs v2

[Slurm currently supports cgroup v1 and v2](https://slurm.schedmd.com/cgroup_v2.html), but there are some limitations with v2, some metrics are not currently fully available on this exporter:

* `memory.max_usage_in_bytes` becomes `memory.peak`, but this is not in any currently released kernel ([torvalds/linux@8e20d4b](https://github.com/torvalds/linux/commit/8e20d4b332660a32e842e20c34cfc3b3456bc6dc)). This is working on 5.14 kernel on EL9.
* `cpuacct.usage_percpu` is exposed via eBPF in kernel 6.6+, but not through cgroupfs although it might be for a future kernel. At the moment, the exporter will get the overall CPU usage of the job and divide it by the number of allocated cores to get an average CPU usage per core.

## Sample

```plaintext
# HELP slurm_job_memory_usage Memory used by a job
# TYPE slurm_job_memory_usage gauge
slurm_job_memory_usage{account="account1",slurmjobid="1",user="user1"} 3.4748194816e+010
slurm_job_memory_usage{account="account2",slurmjobid="2",user="user2"} 3.301916672e+09
slurm_job_memory_usage{account="account3",slurmjobid="3",user="user3"} 1.443540992e+09
slurm_job_memory_usage{account="account4",slurmjobid="4",user="user4"} 6.02486784e+09
# HELP slurm_job_memory_max Maximum memory used by a job
# TYPE slurm_job_memory_max gauge
slurm_job_memory_max{account="account1",slurmjobid="1",user="user1"} 6.8254601216e+010
slurm_job_memory_max{account="account2",slurmjobid="2",user="user2"} 4.294967296e+09
slurm_job_memory_max{account="account3",slurmjobid="3",user="user3"} 1.1140071424e+010
slurm_job_memory_max{account="account4",slurmjobid="4",user="user4"} 9.520877568e+09
# HELP slurm_job_memory_limit Memory limit of a job
# TYPE slurm_job_memory_limit gauge
slurm_job_memory_limit{account="account1",slurmjobid="1",user="user1"} 1.34217728e+011
slurm_job_memory_limit{account="account2",slurmjobid="2",user="user2"} 4.294967296e+09
slurm_job_memory_limit{account="account3",slurmjobid="3",user="user3"} 1.1140071424e+010
slurm_job_memory_limit{account="account4",slurmjobid="4",user="user4"} 3.4359738368e+010
# HELP slurm_job_memory_cache bytes of page cache memory
# TYPE slurm_job_memory_cache gauge
slurm_job_memory_cache{account="account1",slurmjobid="1",user="user1"} 1.8926071808e+010
slurm_job_memory_cache{account="account2",slurmjobid="2",user="user2"} 1.6699392e+09
slurm_job_memory_cache{account="account3",slurmjobid="3",user="user3"} 1.329508352e+09
slurm_job_memory_cache{account="account4",slurmjobid="4",user="user4"} 3.031769088e+09
# HELP slurm_job_memory_rss bytes of anonymous and swap cache memory (includes transparent hugepages).
# TYPE slurm_job_memory_rss gauge
slurm_job_memory_rss{account="account1",slurmjobid="1",user="user1"} 1.5146471424e+010
slurm_job_memory_rss{account="account2",slurmjobid="2",user="user2"} 1.496305664e+09
slurm_job_memory_rss{account="account3",slurmjobid="3",user="user3"} 1.1411456e+07
slurm_job_memory_rss{account="account4",slurmjobid="4",user="user4"} 2.963869696e+09
# HELP slurm_job_memory_rss_huge bytes of anonymous transparent hugepages
# TYPE slurm_job_memory_rss_huge gauge
slurm_job_memory_rss_huge{account="account1",slurmjobid="1",user="user1"} 1.3543407616e+010
slurm_job_memory_rss_huge{account="account2",slurmjobid="2",user="user2"} 5.24288e+08
slurm_job_memory_rss_huge{account="account3",slurmjobid="3",user="user3"} 4.194304e+06
slurm_job_memory_rss_huge{account="account4",slurmjobid="4",user="user4"} 1.43654912e+09
# HELP slurm_job_memory_mapped_file bytes of mapped file (includes tmpfs/shmem)
# TYPE slurm_job_memory_mapped_file gauge
slurm_job_memory_mapped_file{account="account1",slurmjobid="1",user="user1"} 1.0643648512e+010
slurm_job_memory_mapped_file{account="account2",slurmjobid="2",user="user2"} 1.44797696e+08
slurm_job_memory_mapped_file{account="account3",slurmjobid="3",user="user3"} 24576.0
slurm_job_memory_mapped_file{account="account4",slurmjobid="4",user="user4"} 3.0631936e+08
# HELP slurm_job_memory_active_file bytes of file-backed memory on active LRU list
# TYPE slurm_job_memory_active_file gauge
slurm_job_memory_active_file{account="account1",slurmjobid="1",user="user1"} 1.396129792e+09
slurm_job_memory_active_file{account="account2",slurmjobid="2",user="user2"} 1.64872192e+08
slurm_job_memory_active_file{account="account3",slurmjobid="3",user="user3"} 8.0586752e+08
slurm_job_memory_active_file{account="account4",slurmjobid="4",user="user4"} 2.312634368e+09
# HELP slurm_job_memory_inactive_file bytes of file-backed memory on inactive LRU list
# TYPE slurm_job_memory_inactive_file gauge
slurm_job_memory_inactive_file{account="account1",slurmjobid="1",user="user1"} 6.906843136e+09
slurm_job_memory_inactive_file{account="account2",slurmjobid="2",user="user2"} 1.532420096e+09
slurm_job_memory_inactive_file{account="account3",slurmjobid="3",user="user3"} 5.2348928e+08
slurm_job_memory_inactive_file{account="account4",slurmjobid="4",user="user4"} 6.68008448e+08
# HELP slurm_job_memory_unevictable bytes of memory that cannot be reclaimed (mlocked etc)
# TYPE slurm_job_memory_unevictable gauge
slurm_job_memory_unevictable{account="account1",slurmjobid="1",user="user1"} 0.0
slurm_job_memory_unevictable{account="account2",slurmjobid="2",user="user2"} 0.0
slurm_job_memory_unevictable{account="account3",slurmjobid="3",user="user3"} 0.0
slurm_job_memory_unevictable{account="account4",slurmjobid="4",user="user4"} 0.0
# HELP slurm_job_core_usage_total Cpu usage of cores allocated to a job
# TYPE slurm_job_core_usage_total counter
slurm_job_core_usage_total{account="account1",core="12",slurmjobid="1",user="user1"} 3.99304795107903e+014
slurm_job_core_usage_total{account="account2",core="25",slurmjobid="2",user="user2"} 1.8478218205333e+013
slurm_job_core_usage_total{account="account3",core="0",slurmjobid="3",user="user3"} 1.6143782124244e+013
slurm_job_core_usage_total{account="account3",core="1",slurmjobid="3",user="user3"} 1.6290160176392e+013
slurm_job_core_usage_total{account="account4",core="24",slurmjobid="4",user="user4"} 8.40394171634e+011
# HELP slurm_job_process_count Number of processes in a job
# TYPE slurm_job_process_count gauge
slurm_job_process_count{account="account1",slurmjobid="1",user="user1"} 103.0
slurm_job_process_count{account="account2",slurmjobid="2",user="user2"} 6.0
slurm_job_process_count{account="account3",slurmjobid="3",user="user3"} 3.0
slurm_job_process_count{account="account4",slurmjobid="4",user="user4"} 2.0
# HELP slurm_job_threads_count Number of threads in a job
# TYPE slurm_job_threads_count gauge
slurm_job_threads_count{account="account1",slurmjobid="1",state="running",user="user1"} 13.0
slurm_job_threads_count{account="account1",slurmjobid="1",state="sleeping",user="user1"} 163.0
slurm_job_threads_count{account="account2",slurmjobid="2",state="sleeping",user="user2"} 28.0
slurm_job_threads_count{account="account2",slurmjobid="2",state="running",user="user2"} 1.0
slurm_job_threads_count{account="account3",slurmjobid="3",state="sleeping",user="user3"} 7.0
slurm_job_threads_count{account="account4",slurmjobid="4",state="sleeping",user="user4"} 4.0
slurm_job_threads_count{account="account4",slurmjobid="4",state="running",user="user4"} 2.0
# HELP slurm_job_process_usage_total Cpu usage of processes within a job
# TYPE slurm_job_process_usage_total counter
slurm_job_process_usage_total{account="account1",exe="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/python/3.10.13/bin/python3.10",slurmjobid="1",user="user1"} 4.76138055e+06
slurm_job_process_usage_total{account="account2",exe="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/python/3.10.13/bin/python3.10",slurmjobid="2",user="user2"} 535.9900000000001
slurm_job_process_usage_total{account="account4",exe="/home/user4/miniconda3/envs/env/bin/python3.8",slurmjobid="4",user="user4"} 839.03
# HELP slurm_job_memory_total_gpu Memory available on a GPU
# TYPE slurm_job_memory_total_gpu gauge
slurm_job_memory_total_gpu{account="account1",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 4.294967296e+010
slurm_job_memory_total_gpu{account="account2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 4.294967296e+010
slurm_job_memory_total_gpu{account="account4",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 4.294967296e+010
# HELP slurm_job_memory_usage_gpu Memory used by a job on a GPU
# TYPE slurm_job_memory_usage_gpu gauge
slurm_job_memory_usage_gpu{account="account1",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 3.6477861888e+010
slurm_job_memory_usage_gpu{account="account2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 1.103101952e+09
slurm_job_memory_usage_gpu{account="account4",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 8.740929536e+09
# HELP slurm_job_power_gpu Power used by a job on a GPU in mW
# TYPE slurm_job_power_gpu gauge
slurm_job_power_gpu{account="account1",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 71994.0
slurm_job_power_gpu{account="account2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 72894.0
slurm_job_power_gpu{account="account4",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 302176.0
# HELP slurm_job_utilization_gpu Percent of time over the past sample period during which one or more kernels was executing on the GPU.
# TYPE slurm_job_utilization_gpu gauge
slurm_job_utilization_gpu{account="account1",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 1.4786304539431088
slurm_job_utilization_gpu{account="account2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 1.114130389519665
slurm_job_utilization_gpu{account="account4",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 66.45358263087205
# HELP slurm_job_utilization_gpu_memory Percent of time over the past sample period during which global (device) memory was being read or written.
# TYPE slurm_job_utilization_gpu_memory gauge
slurm_job_utilization_gpu_memory{account="account1",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 0.00030990117789661027
slurm_job_utilization_gpu_memory{account="account2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.2555332161063131
slurm_job_utilization_gpu_memory{account="account4",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 28.382148771860084
# HELP slurm_job_sm_occupancy_gpu The ratio of number of warps resident on an SM. (number of resident as a ratio of the theoretical maximum number of warps per elapsed cycle)
# TYPE slurm_job_sm_occupancy_gpu gauge
slurm_job_sm_occupancy_gpu{account="account1",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 0.7616764534250737
slurm_job_sm_occupancy_gpu{account="account2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.3575666628844576
slurm_job_sm_occupancy_gpu{account="account4",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 36.94854688414347
# HELP slurm_job_tensor_gpu The ratio of cycles the tensor (HMMA) pipe is active (off the peak sustained elapsed cycles)
# TYPE slurm_job_tensor_gpu gauge
slurm_job_tensor_gpu{account="account1",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 0.18527105244453848
slurm_job_tensor_gpu{account="account2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.03668124624138651
slurm_job_tensor_gpu{account="account4",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 13.772009921218158
# HELP slurm_job_fp64_gpu Ratio of cycles the fp64 pipe is active
# TYPE slurm_job_fp64_gpu gauge
slurm_job_fp64_gpu{account="account1",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 3.197727320440276e-10
slurm_job_fp64_gpu{account="account2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 3.7664761551282765e-08
slurm_job_fp64_gpu{account="account4",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 0.0
# HELP slurm_job_fp32_gpu Ratio of cycles the fp32 pipe is active
# TYPE slurm_job_fp32_gpu gauge
slurm_job_fp32_gpu{account="account1",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 0.05062633016891654
slurm_job_fp32_gpu{account="account2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.06809155473784152
slurm_job_fp32_gpu{account="account4",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 6.539075793485122
# HELP slurm_job_fp16_gpu Ratio of cycles the fp16 pipe is active
# TYPE slurm_job_fp16_gpu gauge
slurm_job_fp16_gpu{account="account1",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 0.025026645292205888
slurm_job_fp16_gpu{account="account2",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.012512491120997955
slurm_job_fp16_gpu{account="account4",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 0.9619577800276127
# HELP slurm_job_pcie_gpu PCIe tx/rx bytes per second
# TYPE slurm_job_pcie_gpu gauge
slurm_job_pcie_gpu{account="account1",direction="TX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 82729.0
slurm_job_pcie_gpu{account="account1",direction="RX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 63015.0
slurm_job_pcie_gpu{account="account2",direction="TX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 3.4488889e+07
slurm_job_pcie_gpu{account="account2",direction="RX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 2.2912168e+07
slurm_job_pcie_gpu{account="account4",direction="TX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 1.1561547e+07
slurm_job_pcie_gpu{account="account4",direction="RX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 5.1336384e+07
# HELP slurm_job_nvlink_gpu Nvlink tx/rx bytes per second
# TYPE slurm_job_nvlink_gpu gauge
slurm_job_nvlink_gpu{account="account1",direction="TX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 0.0
slurm_job_nvlink_gpu{account="account1",direction="RX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="1",user="user1"} 0.0
slurm_job_nvlink_gpu{account="account2",direction="TX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.0
slurm_job_nvlink_gpu{account="account2",direction="RX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="2",user="user2"} 0.0
slurm_job_nvlink_gpu{account="account4",direction="TX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 0.0
slurm_job_nvlink_gpu{account="account4",direction="RX",gpu="0",gpu_type="NVIDIA A100-SXM4-40GB",slurmjobid="4",user="user4"} 0.0
```
