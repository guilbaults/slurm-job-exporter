# Slurm-job-exporter
Prometheus exporter for the stats in the cgroup accounting with slurm. This will also collect stats of a job using NVIDIA GPUs.

## Requirements
Python 3 with the following modules:

* `prometheus_client`
* `nvidia-ml-py`

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
slurm_job_memory_usage{slurmjobid="16767964",user="user1"} 5.673777152e+010
slurm_job_memory_usage{slurmjobid="16761306",user="user2"} 1.973624832e+09
slurm_job_memory_usage{slurmjobid="16761305",user="user2"} 2.264969216e+09
slurm_job_memory_usage{slurmjobid="16761228",user="user2"} 2.884288512e+09
# HELP slurm_job_memory_max Maximum memory used by a job
# TYPE slurm_job_memory_max gauge
slurm_job_memory_max{slurmjobid="16767964",user="user1"} 5.6738013184e+010
slurm_job_memory_max{slurmjobid="16761306",user="user2"} 2.049892352e+09
slurm_job_memory_max{slurmjobid="16761305",user="user2"} 2.264997888e+09
slurm_job_memory_max{slurmjobid="16761228",user="user2"} 2.980458496e+09
# HELP slurm_job_memory_limit Memory limit of a job
# TYPE slurm_job_memory_limit gauge
slurm_job_memory_limit{slurmjobid="16767964",user="user1"} 6.8719476736e+010
slurm_job_memory_limit{slurmjobid="16761306",user="user2"} 1.073741824e+010
slurm_job_memory_limit{slurmjobid="16761305",user="user2"} 1.073741824e+010
slurm_job_memory_limit{slurmjobid="16761228",user="user2"} 1.073741824e+010
# HELP slurm_job_core_usage_total Cpu usage of cores allocated to a job
# TYPE slurm_job_core_usage_total counter
slurm_job_core_usage_total{core="18",slurmjobid="16767964",user="user1"} 1.9894422351762e+013
slurm_job_core_usage_total{core="20",slurmjobid="16767964",user="user1"} 2.0269313668611e+013
slurm_job_core_usage_total{core="22",slurmjobid="16767964",user="user1"} 2.0005359820476e+013
slurm_job_core_usage_total{core="24",slurmjobid="16767964",user="user1"} 1.9959527685329e+013
slurm_job_core_usage_total{core="26",slurmjobid="16767964",user="user1"} 1.9928711737765e+013
slurm_job_core_usage_total{core="28",slurmjobid="16767964",user="user1"} 1.9930795813539e+013
slurm_job_core_usage_total{core="30",slurmjobid="16767964",user="user1"} 1.9981840929731e+013
slurm_job_core_usage_total{core="32",slurmjobid="16767964",user="user1"} 1.999003786346e+013
slurm_job_core_usage_total{core="1",slurmjobid="16761306",user="user2"} 1.38636676827e+011
slurm_job_core_usage_total{core="3",slurmjobid="16761306",user="user2"} 1.11455618511e+011
slurm_job_core_usage_total{core="5",slurmjobid="16761306",user="user2"} 8.5467479122e+010
slurm_job_core_usage_total{core="7",slurmjobid="16761306",user="user2"} 8.003129745e+010
slurm_job_core_usage_total{core="9",slurmjobid="16761306",user="user2"} 8.2835983178e+010
slurm_job_core_usage_total{core="11",slurmjobid="16761306",user="user2"} 8.0290729258e+010
slurm_job_core_usage_total{core="13",slurmjobid="16761306",user="user2"} 7.559852192e+010
slurm_job_core_usage_total{core="15",slurmjobid="16761306",user="user2"} 1.20074698286e+011
slurm_job_core_usage_total{core="17",slurmjobid="16761306",user="user2"} 1.38848698254e+011
slurm_job_core_usage_total{core="19",slurmjobid="16761306",user="user2"} 1.31007531893e+011
slurm_job_core_usage_total{core="0",slurmjobid="16761305",user="user2"} 7.3694637403e+010
slurm_job_core_usage_total{core="2",slurmjobid="16761305",user="user2"} 1.53043851085e+011
slurm_job_core_usage_total{core="4",slurmjobid="16761305",user="user2"} 1.35456747954e+011
slurm_job_core_usage_total{core="6",slurmjobid="16761305",user="user2"} 1.37753030743e+011
slurm_job_core_usage_total{core="8",slurmjobid="16761305",user="user2"} 1.18177450244e+011
slurm_job_core_usage_total{core="10",slurmjobid="16761305",user="user2"} 8.578527458e+010
slurm_job_core_usage_total{core="12",slurmjobid="16761305",user="user2"} 7.1694327955e+010
slurm_job_core_usage_total{core="14",slurmjobid="16761305",user="user2"} 1.04234670793e+011
slurm_job_core_usage_total{core="34",slurmjobid="16761305",user="user2"} 7.9575309731e+010
slurm_job_core_usage_total{core="36",slurmjobid="16761305",user="user2"} 8.0867897065e+010
slurm_job_core_usage_total{core="16",slurmjobid="16761228",user="user2"} 3.6710739094e+010
slurm_job_core_usage_total{core="21",slurmjobid="16761228",user="user2"} 1.909029415447e+012
slurm_job_core_usage_total{core="23",slurmjobid="16761228",user="user2"} 1.510500678372e+012
slurm_job_core_usage_total{core="25",slurmjobid="16761228",user="user2"} 1.360207129614e+012
slurm_job_core_usage_total{core="27",slurmjobid="16761228",user="user2"} 1.624474633884e+012
slurm_job_core_usage_total{core="29",slurmjobid="16761228",user="user2"} 1.600589392158e+012
slurm_job_core_usage_total{core="31",slurmjobid="16761228",user="user2"} 1.98745099097e+012
slurm_job_core_usage_total{core="33",slurmjobid="16761228",user="user2"} 1.816077051646e+012
slurm_job_core_usage_total{core="35",slurmjobid="16761228",user="user2"} 1.790231247047e+012
slurm_job_core_usage_total{core="38",slurmjobid="16761228",user="user2"} 2.5595053422e+010
# HELP slurm_job_memory_usage_gpu Memory used by a job on a GPU
# TYPE slurm_job_memory_usage_gpu gauge
slurm_job_memory_usage_gpu{gpu="3",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16767964",user="user1"} 1.65937152e+09
slurm_job_memory_usage_gpu{gpu="1",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761306",user="user2"} 1.5498477568e+010
slurm_job_memory_usage_gpu{gpu="0",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761305",user="user2"} 1.5498477568e+010
slurm_job_memory_usage_gpu{gpu="2",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761228",user="user2"} 1.5498477568e+010
# HELP slurm_job_power_gpu Power used by a job on a GPU in mW
# TYPE slurm_job_power_gpu gauge
slurm_job_power_gpu{gpu="3",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16767964",user="user1"} 53257.0
slurm_job_power_gpu{gpu="1",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761306",user="user2"} 193506.0
slurm_job_power_gpu{gpu="0",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761305",user="user2"} 146698.0
slurm_job_power_gpu{gpu="2",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761228",user="user2"} 104854.0
# HELP slurm_job_utilization_gpu Percent of time over the past sample period during which one or more kernels was executing on the GPU.
# TYPE slurm_job_utilization_gpu gauge
slurm_job_utilization_gpu{gpu="3",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16767964",user="user1"} 28.0
slurm_job_utilization_gpu{gpu="1",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761306",user="user2"} 55.0
slurm_job_utilization_gpu{gpu="0",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761305",user="user2"} 52.0
slurm_job_utilization_gpu{gpu="2",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761228",user="user2"} 22.0
# HELP slurm_job_memory_utilization_gpu Percent of time over the past sample period during which global (device) memory was being read or written.
# TYPE slurm_job_memory_utilization_gpu gauge
slurm_job_memory_utilization_gpu{gpu="3",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16767964",user="user1"} 9.0
slurm_job_memory_utilization_gpu{gpu="1",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761306",user="user2"} 28.0
slurm_job_memory_utilization_gpu{gpu="0",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761305",user="user2"} 30.0
slurm_job_memory_utilization_gpu{gpu="2",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761228",user="user2"} 10.0
# HELP slurm_job_pcie_gpu PCIe throughput in KB/s
# TYPE slurm_job_pcie_gpu gauge
slurm_job_pcie_gpu{direction="TX",gpu="3",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16767964",user="user1"} 0.0
slurm_job_pcie_gpu{direction="RX",gpu="3",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16767964",user="user1"} 0.0
slurm_job_pcie_gpu{direction="TX",gpu="1",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761306",user="user2"} 16000.0
slurm_job_pcie_gpu{direction="RX",gpu="1",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761306",user="user2"} 21000.0
slurm_job_pcie_gpu{direction="TX",gpu="0",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761305",user="user2"} 15000.0
slurm_job_pcie_gpu{direction="RX",gpu="0",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761305",user="user2"} 67000.0
slurm_job_pcie_gpu{direction="TX",gpu="2",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761228",user="user2"} 7000.0
slurm_job_pcie_gpu{direction="RX",gpu="2",gpu_type="Tesla V100-SXM2-16GB",slurmjobid="16761228",user="user2"} 22000.0
```
