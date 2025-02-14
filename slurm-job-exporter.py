import glob
import argparse
import subprocess
import re
import sys
import psutil
import os
from functools import lru_cache
from wsgiref.simple_server import make_server, WSGIRequestHandler
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from prometheus_client import make_wsgi_app


GPU_UUID_RE = re.compile('(GPU|MIG)-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})')


@lru_cache(maxsize=100)
def get_username(uid):
    """
    Convert a numerical uid to a username
    """
    command = ['/usr/bin/id', '--name', '--user', '{}'.format(uid)]
    return subprocess.check_output(command).strip().decode()


def cgroup_processes(job_dir):
    """
    Find all the PIDs for a cgroup of a job
    """
    procs = []
    res_uid = -1
    for (path, _, _) in os.walk(job_dir):
        with open(os.path.join(path, "cgroup.procs"), 'r') as fprocs:
            for proc in fprocs.readlines():
                pid = int(proc)
                try:
                    ps = psutil.Process(pid)
                    uid = ps.uids().real
                    if uid != 0:
                        res_uid = uid
                        procs.append(pid)
                except psutil.NoSuchProcess:
                    pass
    return res_uid, procs


def split_range(range_str):
    """"
    split a range such as "0-1,3,5,10-13"
    to 0,1,3,5,10,11,12,13
    """
    ranges = []
    for sub in range_str.split(','):
        if '-' in sub:
            subrange = sub.split('-')
            for i in range(int(subrange[0]), int(subrange[1]) + 1):
                ranges.append(i)
        else:
            ranges.append(int(sub))
    return ranges


def get_env(pid):
    """
    Return the environment variables of a process
    """
    try:
        ps = psutil.Process(pid)
        return ps.environ()
    except psutil.NoSuchProcess:
        raise ValueError("Could not get environment for {}".format(pid))


def cgroup_gpus(job_dir, cgroups):
    if cgroups == 1:
        task_file = os.path.join(job_dir, "tasks")
    else:
        cgroup_path = os.path.join(job_dir, "gpu_probe")
        # This will create a new cgroup under the root of the job.
        # This is required for v2 since we can only add tasks to leaf cgroups
        os.mkdir(cgroup_path)
        task_file = os.path.join(cgroup_path, "cgroup.procs")
    try:
        res = subprocess.check_output(["get_gpus.sh", task_file]).strip().decode()
    except FileNotFoundError:
        # This is most likely because nvidia-smi is not on the machine
        return []
    finally:
        if cgroups == 2:
            # We can remove a cgroup if no tasks are remaining inside
            os.rmdir(cgroup_path)

    gpus = []

    mig = 'MIG' in res
    for line in res.split('\n'):
        m = GPU_UUID_RE.search(line)
        if mig and m and m.group(1) == 'MIG':
            gpus.append((None, m.group()))
        elif not mig and m and m.group(1) == 'GPU':
            gpu = str(line.split()[1].rstrip(':'))
            gpus.append((gpu, m.group()))
    return gpus


class SlurmJobCollector(object):
    """
    Used by a WSGI application to collect and return stats about currently
    running slurm jobs on a node. This is using the stats from the cgroups
    created by Slurm.
    """
    def __init__(self, dcgm_update_interval=10):
        """
        Args:
            dcgm_update_interval (int, optional): Pooling interval in seconds used by DCGM. Defaults to 10.
        """
        # Will be auto detected by the exporter
        self.MONITOR_DCGM = False
        self.MONITOR_PYNVML = False
        self.UNSUPPORTED_FEATURES = []
        for proc in psutil.process_iter():
            if proc.name() == 'nv-hostengine':
                # DCGM is running on this host
                # Load DCGM bindings from the RPM
                sys.path.insert(0, '/usr/local/dcgm/bindings/python3/')

                try:
                    import pydcgm
                    import dcgm_fields
                    import dcgm_structs

                    self.handle = pydcgm.DcgmHandle(None, 'localhost')
                    self.group = pydcgm.DcgmGroup(self.handle, groupName="slurm-job-exporter", groupType=dcgm_structs.DCGM_GROUP_DEFAULT_INSTANCES)

                    if len(self.group.GetEntities()) == 0:
                        # No MIG, switch to default group
                        self.group.Delete()
                        self.group = pydcgm.DcgmGroup(self.handle, groupName="slurm-job-exporter", groupType=dcgm_structs.DCGM_GROUP_DEFAULT)

                    # https://github.com/NVIDIA/gpu-monitoring-tools/blob/master/bindings/go/dcgm/dcgm_fields.h
                    self.fieldIds_dict = {
                        dcgm_fields.DCGM_FI_DEV_NAME: 'name',
                        dcgm_fields.DCGM_FI_DEV_UUID: 'uuid',
                        dcgm_fields.DCGM_FI_DEV_CUDA_VISIBLE_DEVICES_STR: 'cuda_visible_devices_str',
                        dcgm_fields.DCGM_FI_DEV_POWER_USAGE: 'power_usage',
                        dcgm_fields.DCGM_FI_DEV_FB_USED: 'fb_used',
                        dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE: 'fp64_active',
                        dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE: 'fp32_active',
                        dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE: 'fp16_active',
                        dcgm_fields.DCGM_FI_PROF_SM_ACTIVE: 'sm_active',
                        dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY: 'sm_occupancy',
                        dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE: 'tensor_active',
                        dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE: 'dram_active',
                        dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES: 'pcie_tx_bytes',
                        dcgm_fields.DCGM_FI_PROF_PCIE_RX_BYTES: 'pcie_rx_bytes',
                        dcgm_fields.DCGM_FI_PROF_NVLINK_TX_BYTES: 'nvlink_tx_bytes',
                        dcgm_fields.DCGM_FI_PROF_NVLINK_RX_BYTES: 'nvlink_rx_bytes',
                    }

                    for gpu_id in pydcgm.DcgmSystemDiscovery(self.handle).GetAllSupportedGpuIds():
                        device = pydcgm.dcgm_agent.dcgmGetDeviceAttributes(self.handle.handle, gpu_id)
                        name = device.identifiers.deviceName
                        print('Detected gpu {} with ID {}'.format(name, gpu_id))

                    self.field_group = pydcgm.DcgmFieldGroup(self.handle, name="slurm-job-exporter-fg", fieldIds=list(self.fieldIds_dict.keys()))

                    try:
                        # try watching with lp64 features
                        self.group.samples.WatchFields(self.field_group, dcgm_update_interval * 1000 * 1000, dcgm_update_interval * 2.0, 0)
                    except dcgm_structs.DCGMError_NotSupported:
                        # slightly kludgy: recreate group - without fp64
                        self.field_group.Delete()
                        del self.fieldIds_dict[dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE]
                        self.UNSUPPORTED_FEATURES.append('fp64')
                        self.field_group = pydcgm.DcgmFieldGroup(self.handle, name="slurm-job-exporter-fg", fieldIds=list(self.fieldIds_dict.keys()))

                        # try watching without lp64 features
                        self.group.samples.WatchFields(self.field_group, dcgm_update_interval * 1000 * 1000, dcgm_update_interval * 2.0, 0)

                        print('Disabled fp64 metrics as an installed gpu does not support it')

                    self.handle.GetSystem().UpdateAllFields(True)

                    print('Monitoring GPUs with DCGM with an update interval of {} seconds'.format(dcgm_update_interval))
                    self.MONITOR_DCGM = True
                    self.MONITOR_PYNVML = False
                except ImportError:
                    self.MONITOR_DCGM = False

        # using nvml as a fallback for DCGM
        if self.MONITOR_DCGM is False:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.MONITOR_PYNVML = True
                print('Monitoring GPUs with pynvml')
                self.pynvml = pynvml
            except ImportError:
                self.MONITOR_PYNVML = False
            except pynvml.NVMLError_LibraryNotFound:
                self.MONITOR_PYNVML = False
            except pynvml.NVMLError_DriverNotLoaded:
                self.MONITOR_PYNVML = False

    def GetLatestGpuValuesAsDict(self):
        gpus = {}
        data = self.group.samples.GetLatest_v2(self.field_group).values
        for k in data.keys():
            for v in data[k].keys():
                data_dict = {}
                for metric_id in data[k][v].keys():
                    data_dict[self.fieldIds_dict[metric_id]] = data[k][v][metric_id].values[0].value
                gpus[data_dict['uuid']] = data_dict
        return gpus

    def collect(self):
        """
        Run a collection cycle and update exported stats
        """
        gauge_memory_usage = GaugeMetricFamily(
            'slurm_job_memory_usage', 'Memory used by a job',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_max = GaugeMetricFamily(
            'slurm_job_memory_max', 'Maximum memory used by a job',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_limit = GaugeMetricFamily(
            'slurm_job_memory_limit', 'Memory limit of a job',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_cache = GaugeMetricFamily(
            'slurm_job_memory_cache', 'bytes of page cache memory',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_rss = GaugeMetricFamily(
            'slurm_job_memory_rss',
            'bytes of anonymous and swap cache memory (includes transparent hugepages).',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_rss_huge = GaugeMetricFamily(
            'slurm_job_memory_rss_huge',
            'bytes of anonymous transparent hugepages',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_mapped_file = GaugeMetricFamily(
            'slurm_job_memory_mapped_file',
            'bytes of mapped file (includes tmpfs/shmem)',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_active_file = GaugeMetricFamily(
            'slurm_job_memory_active_file',
            'bytes of file-backed memory on active LRU list',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_inactive_file = GaugeMetricFamily(
            'slurm_job_memory_inactive_file',
            'bytes of file-backed memory on inactive LRU list',
            labels=['user', 'account', 'slurmjobid'])
        gauge_memory_unevictable = GaugeMetricFamily(
            'slurm_job_memory_unevictable',
            'bytes of memory that cannot be reclaimed (mlocked etc)',
            labels=['user', 'account', 'slurmjobid'])

        counter_core_usage = CounterMetricFamily(
            'slurm_job_core_usage', 'Cpu usage of cores allocated to a job',
            labels=['user', 'account', 'slurmjobid', 'core'])

        gauge_process_count = GaugeMetricFamily(
            'slurm_job_process_count', 'Number of processes in a job',
            labels=['user', 'account', 'slurmjobid'])
        gauge_threads_count = GaugeMetricFamily(
            'slurm_job_threads_count', 'Number of threads in a job',
            labels=['user', 'account', 'slurmjobid', 'state'])

        counter_process_usage = CounterMetricFamily(
            'slurm_job_process_usage', 'Cpu usage of processes within a job',
            labels=['user', 'account', 'slurmjobid', 'exe'])

        if self.MONITOR_PYNVML or self.MONITOR_DCGM:
            # pynvml is used as a fallback for DCGM, both can collect GPU stats
            gauge_memory_usage_gpu = GaugeMetricFamily(
                'slurm_job_memory_usage_gpu', 'Memory used by a job on a GPU',
                labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type'])
            gauge_power_gpu = GaugeMetricFamily(
                'slurm_job_power_gpu', 'Power used by a job on a GPU in mW',
                labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type'])
            gauge_utilization_gpu = GaugeMetricFamily(
                'slurm_job_utilization_gpu',
                'Percent of time over the past sample period during which \
one or more kernels was executing on the GPU.',
                labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type'])
            gauge_memory_utilization_gpu = GaugeMetricFamily(
                'slurm_job_utilization_gpu_memory',
                'Percent of time over the past sample period during which \
global (device) memory was being read or written.',
                labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type'])

        if self.MONITOR_DCGM:
            # DCGM have additional metrics for GPU
            gauge_sm_occupancy_gpu = GaugeMetricFamily(
                'slurm_job_sm_occupancy_gpu',
                'The ratio of number of warps resident on an SM. \
(number of resident as a ratio of the theoretical maximum number of warps \
per elapsed cycle)',
                labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type'])
            gauge_tensor_gpu = GaugeMetricFamily(
                'slurm_job_tensor_gpu',
                'The ratio of cycles the tensor (HMMA) pipe is active \
(off the peak sustained elapsed cycles)',
                labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type'])
            if 'fp64' not in self.UNSUPPORTED_FEATURES:
                gauge_fp64_gpu = GaugeMetricFamily(
                    'slurm_job_fp64_gpu',
                    'Ratio of cycles the fp64 pipe is active',
                    labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type'])
            gauge_fp32_gpu = GaugeMetricFamily(
                'slurm_job_fp32_gpu',
                'Ratio of cycles the fp32 pipe is active',
                labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type'])
            gauge_fp16_gpu = GaugeMetricFamily(
                'slurm_job_fp16_gpu',
                'Ratio of cycles the fp16 pipe is active',
                labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type'])
            gauge_nvlink_gpu = GaugeMetricFamily(
                'slurm_job_nvlink_gpu', 'Nvlink tx/rx bytes per second',
                labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type', 'direction'])
            gauge_pcie_gpu = GaugeMetricFamily(
                'slurm_job_pcie_gpu', 'PCIe tx/rx bytes per second',
                labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type', 'direction'])

        if os.path.exists("/sys/fs/cgroup/memory"):
            cgroups = 1  # we are running cgroups v1
        else:
            cgroups = 2  # we are running cgroups v2

        if cgroups == 1:
            jobs_glob = "/sys/fs/cgroup/memory/slurm/uid_*/job_*"
        else:
            jobs_glob = "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_*"
        for job_dir in glob.glob(jobs_glob):
            job = job_dir.split('/')[-1].split('_')[1]
            uid, procs = cgroup_processes(job_dir)
            if len(procs) == 0:
                continue

            # Job is alive, we can get the stats
            user = get_username(uid)
            gpu_set = set()
            if self.MONITOR_PYNVML or self.MONITOR_DCGM:
                if cgroups == 1:
                    gpu_dir = "/sys/fs/cgroup/devices/slurm/uid_{}/job_{}".format(uid, job)
                else:
                    gpu_dir = job_dir
                gpu_set.update(cgroup_gpus(gpu_dir, cgroups))

            for proc in procs:
                # get the SLURM_JOB_ACCOUNT
                try:
                    envs = get_env(proc)
                except ValueError:
                    # Process does not have an environment, its probably gone
                    continue
                if 'SLURM_JOB_ACCOUNT' in envs:
                    account = envs['SLURM_JOB_ACCOUNT']
                    break
            else:
                # Could not find the env variables, slurm_adopt only fill the jobid
                account = "error"

            with open(os.path.join(job_dir, ('memory.usage_in_bytes' if cgroups == 1 else 'memory.current')), 'r') as f_usage:
                gauge_memory_usage.add_metric([user, account, job], int(f_usage.read()))
            try:
                with open(os.path.join(job_dir, ('memory.max_usage_in_bytes' if cgroups == 1 else 'memory.peak')), 'r') as f_max:
                    gauge_memory_max.add_metric([user, account, job], int(f_max.read()))
            except FileNotFoundError:
                # 'memory.peak' is only available in kernel 6.8+
                pass

            with open(os.path.join(job_dir, ('memory.limit_in_bytes' if cgroups == 1 else 'memory.max')), 'r') as f_limit:
                gauge_memory_limit.add_metric([user, account, job], int(f_limit.read()))

            with open(os.path.join(job_dir, 'memory.stat'), 'r') as f_stats:
                stats = dict(line.split() for line in f_stats.readlines())
            if cgroups == 1:
                gauge_memory_cache.add_metric(
                    [user, account, job], int(stats['total_cache']))
                gauge_memory_rss.add_metric(
                    [user, account, job], int(stats['total_rss']))
                gauge_memory_rss_huge.add_metric(
                    [user, account, job], int(stats['total_rss_huge']))
                gauge_memory_mapped_file.add_metric(
                    [user, account, job], int(stats['total_mapped_file']))
                gauge_memory_active_file.add_metric(
                    [user, account, job], int(stats['total_active_file']))
                gauge_memory_inactive_file.add_metric(
                    [user, account, job], int(stats['total_inactive_file']))
                gauge_memory_unevictable.add_metric(
                    [user, account, job], int(stats['total_unevictable']))
            else:
                gauge_memory_cache.add_metric(
                    [user, account, job], int(stats['file']))
                gauge_memory_rss.add_metric(
                    [user, account, job],
                    int(stats['anon']) + int(stats['swapcached']))
                gauge_memory_rss_huge.add_metric(
                    [user, account, job], int(stats['anon_thp']))
                gauge_memory_mapped_file.add_metric(
                    [user, account, job],
                    int(stats['file_mapped']) + int(stats['shmem']))
                gauge_memory_active_file.add_metric(
                    [user, account, job], int(stats['active_file']))
                gauge_memory_inactive_file.add_metric(
                    [user, account, job], int(stats['inactive_file']))
                gauge_memory_unevictable.add_metric(
                    [user, account, job], int(stats['unevictable']))

            # get the allocated cores
            if cgroups == 1:
                cpuset_path = '/sys/fs/cgroup/cpuset/slurm/uid_{}/job_{}/cpuset.effective_cpus'.format(uid, job)
            else:
                cpuset_path = os.path.join(job_dir, 'cpuset.cpus.effective')

            with open(cpuset_path, 'r') as f_cores:
                cores = split_range(f_cores.read())

            if cgroups == 1:
                # There is no equivalent to this in cgroups v2
                with open('/sys/fs/cgroup/cpu,cpuacct/slurm/uid_{}/job_{}/cpuacct.usage_percpu'.format(uid, job), 'r') as f_usage:
                    cpu_usages = f_usage.read().split()
                    for core in cores:
                        counter_core_usage.add_metric([user, account, job, str(core)],
                                                      int(cpu_usages[core]))

            processes = 0
            tasks_state = {}
            for proc in procs:
                try:
                    p = psutil.Process(proc)
                    cmdline = p.cmdline()
                except psutil.NoSuchProcess:
                    continue
                if len(cmdline) == 0:
                    # sometimes the cmdline is empty, we don't want to count it
                    continue
                if cmdline[0] == '/bin/bash':
                    if len(cmdline) > 1:
                        if '/var/spool' in cmdline[1] and 'slurm_script' in cmdline[1]:
                            # This is the bash script of the job, we don't want to count it
                            continue
                processes += 1

                for t in p.threads():
                    try:
                        pt = psutil.Process(t.id)
                    except psutil.NoSuchProcess:
                        # The thread disappeared between the time we got the list and now
                        continue
                    pt_status = pt.status()
                    if pt_status in tasks_state:
                        tasks_state[pt_status] += 1
                    else:
                        tasks_state[pt_status] = 1

            for status in tasks_state.keys():
                gauge_threads_count.add_metric([user, account, job, status], tasks_state[status])
            gauge_process_count.add_metric([user, account, job], processes)

            processes_sum = {}
            for proc in procs:
                # get the counter_process_usage data
                try:
                    p = psutil.Process(proc)
                    with p.oneshot():
                        exe = p.exe()
                    if os.path.basename(exe) in ['ssh', 'sshd', 'bash', 'srun']:
                        # We don't want to count them
                        continue
                    else:
                        t = p.cpu_times().user + p.cpu_times().system + p.cpu_times().children_user + p.cpu_times().children_system
                        if exe in processes_sum:
                            processes_sum[exe] += t
                        else:
                            processes_sum[exe] = t
                except psutil.NoSuchProcess:
                    continue

            # we only count the processes that used more than 60 seconds of CPU
            processes_sum_filtered = processes_sum.copy()
            for exe in processes_sum.keys():
                if processes_sum[exe] < 60:
                    del processes_sum_filtered[exe]

            for exe in processes_sum_filtered.keys():
                counter_process_usage.add_metric([user, account, job, exe], processes_sum_filtered[exe])

                if self.MONITOR_PYNVML:
                    for gpu in gpu_set:
                        gpu = int(gpu[0])
                        handle = self.pynvml.nvmlDeviceGetHandleByIndex(gpu)
                        name = self.pynvml.nvmlDeviceGetName(handle)
                        if type(name) is str:
                            gpu_type = self.pynvml.nvmlDeviceGetName(handle)
                        else:
                            gpu_type = self.pynvml.nvmlDeviceGetName(handle).decode()
                        gauge_memory_usage_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type],
                            int(self.pynvml.nvmlDeviceGetMemoryInfo(handle).used))
                        try:
                            gauge_power_gpu.add_metric(
                                [user, account, job, str(gpu), gpu_type],
                                self.pynvml.nvmlDeviceGetPowerUsage(handle))
                        except self.pynvml.NVMLError_NotSupported:
                            pass
                        utils = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gauge_utilization_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type], utils.gpu)
                        gauge_memory_utilization_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type], utils.memory)

                if self.MONITOR_DCGM:
                    dcgm_data = self.GetLatestGpuValuesAsDict()
                    for gpu_tuple in gpu_set:
                        if gpu_tuple[0] is None:
                            # MIG, use the UUID of the GPU
                            gpu = gpu_tuple[1]
                        else:
                            # Full GPU, can be with dcgm or pynvml, use the gpu number
                            gpu = gpu_tuple[0]
                        gpu_uuid = gpu_tuple[1]
                        gpu_type = dcgm_data[gpu_uuid]['name']
                        # Converting DCGM data to the same format as NVML and reusing the same metrics
                        gauge_memory_usage_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type],
                            int(dcgm_data[gpu_uuid]['fb_used']) * 1024 * 1024)  # convert to bytes
                        gauge_power_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type],
                            dcgm_data[gpu_uuid]['power_usage'] * 1000)  # convert to mW
                        gauge_utilization_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type],
                            dcgm_data[gpu_uuid]['sm_active'] * 100)  # convert to %
                        gauge_memory_utilization_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type],
                            dcgm_data[gpu_uuid]['dram_active'] * 100)  # convert to %

                        # Convert to % to keep the same format as NVML
                        gauge_sm_occupancy_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type],
                            dcgm_data[gpu_uuid]['sm_occupancy'] * 100)
                        gauge_tensor_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type],
                            dcgm_data[gpu_uuid]['tensor_active'] * 100)
                        if 'fp64' not in self.UNSUPPORTED_FEATURES:
                            gauge_fp64_gpu.add_metric(
                                [user, account, job, str(gpu), gpu_type],
                                dcgm_data[gpu_uuid]['fp64_active'] * 100)
                        gauge_fp32_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type],
                            dcgm_data[gpu_uuid]['fp32_active'] * 100)
                        gauge_fp16_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type],
                            dcgm_data[gpu_uuid]['fp16_active'] * 100)

                        gauge_pcie_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type, 'TX'],
                            dcgm_data[gpu_uuid]['pcie_tx_bytes'])
                        gauge_pcie_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type, 'RX'],
                            dcgm_data[gpu_uuid]['pcie_rx_bytes'])
                        gauge_nvlink_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type, 'TX'],
                            dcgm_data[gpu_uuid]['nvlink_tx_bytes'])
                        gauge_nvlink_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type, 'RX'],
                            dcgm_data[gpu_uuid]['nvlink_rx_bytes'])

        yield gauge_memory_usage
        yield gauge_memory_max
        yield gauge_memory_limit
        yield gauge_memory_cache
        yield gauge_memory_rss
        yield gauge_memory_rss_huge
        yield gauge_memory_mapped_file
        yield gauge_memory_active_file
        yield gauge_memory_inactive_file
        yield gauge_memory_unevictable
        yield counter_core_usage
        yield gauge_process_count
        yield gauge_threads_count
        yield counter_process_usage

        if self.MONITOR_PYNVML or self.MONITOR_DCGM:
            yield gauge_memory_usage_gpu
            yield gauge_power_gpu
            yield gauge_utilization_gpu
            yield gauge_memory_utilization_gpu
        if self.MONITOR_DCGM:
            yield gauge_sm_occupancy_gpu
            yield gauge_tensor_gpu
            if 'fp64' not in self.UNSUPPORTED_FEATURES:
                yield gauge_fp64_gpu
            yield gauge_fp32_gpu
            yield gauge_fp16_gpu
            yield gauge_pcie_gpu
            yield gauge_nvlink_gpu


class NoLoggingWSGIRequestHandler(WSGIRequestHandler):
    """
    Class to remove logging of WSGI
    """
    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Promtheus exporter for jobs running with Slurm \
within a cgroup')
    PARSER.add_argument(
        '--port',
        type=int,
        default=9798,
        help='Collector http port, default is 9798')
    PARSER.add_argument(
        '--dcgm-update-interval',
        type=int,
        default=10,
        help='DCGM update interval in seconds, default is 10')
    ARGS = PARSER.parse_args()

    APP = make_wsgi_app(SlurmJobCollector(dcgm_update_interval=ARGS.dcgm_update_interval))
    HTTPD = make_server('', ARGS.port, APP,
                        handler_class=NoLoggingWSGIRequestHandler)
    HTTPD.serve_forever()
