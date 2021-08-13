import glob
import argparse
import subprocess
import re
from functools import lru_cache
from wsgiref.simple_server import make_server, WSGIRequestHandler
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from prometheus_client import make_wsgi_app

try:
    import pynvml
    pynvml.nvmlInit()
    MONITOR_GPU = True
    print('Monitoring GPUs')
except ImportError:
    MONITOR_GPU = False
except pynvml.NVMLError_DriverNotLoaded:
    MONITOR_GPU = False


@lru_cache(maxsize=100)
def get_username(uid):
    """
    Convert a numerical uid to a username
    """
    command = ['/usr/bin/id', '--name', '--user', '{}'.format(uid)]
    return subprocess.check_output(command).strip().decode()


def cgroup_processes(uid, job):
    """
    Find all the PIDs for a cgroup of a user+job
    """
    procs = []
    step_g = '/sys/fs/cgroup/memory/slurm/uid_{}/job_{}/step_*'
    for step in glob.glob(step_g.format(uid, job)):
        cgroup = '/sys/fs/cgroup/memory/slurm/uid_{}/job_{}/{}/task_*'.format(
            uid, job, step.split('/')[-1])
        for process_file in glob.glob(cgroup):
            with open(process_file + '/tasks', 'r') as stats:
                for proc in stats.readlines():
                    procs.append(proc.strip())
    return procs


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
    environments = {}
    with open('/proc/{}/environ'.format(pid), 'r') as env_f:
        for env in env_f.read().split('\000'):
            r_env = re.match(r'(.*)=(.*)', env)
            if r_env:
                environments[r_env.group(1)] = r_env.group(2)
    return environments


class SlurmJobCollector(object):
    """
    Used by a WSGI application to collect and return stats about currently
    running slurm jobs on a node. This is using the stats from the cgroups
    created by Slurm.
    """
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

        if MONITOR_GPU:
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
            gauge_pcie_gpu = GaugeMetricFamily(
                'slurm_job_pcie_gpu', 'PCIe throughput in KB/s',
                labels=['user', 'account', 'slurmjobid', 'gpu', 'gpu_type', 'direction'])

        for uid_dir in glob.glob("/sys/fs/cgroup/memory/slurm/uid_*"):
            uid = uid_dir.split('/')[-1].split('_')[1]
            job_path = "/sys/fs/cgroup/memory/slurm/uid_{}/job_*".format(uid)
            for job_dir in glob.glob(job_path):
                job = job_dir.split('/')[-1].split('_')[1]
                mem_path = '/sys/fs/cgroup/memory/slurm/uid_{}/job_{}/'.format(
                    uid, job)
                procs = cgroup_processes(uid, job)
                if len(procs) == 0:
                    continue

                # Job is alive, we can get the stats
                user = get_username(uid)
                gpu_set = set()
                for proc in procs:
                    envs = get_env(proc)
                    if 'SLURM_JOB_ACCOUNT' in envs:
                        account = envs['SLURM_JOB_ACCOUNT']

                        if MONITOR_GPU:
                            if 'SLURM_JOB_GPUS' in envs:
                                gpus = envs['SLURM_JOB_GPUS'].split(',')
                            elif 'SLURM_STEP_GPUS' in envs:
                                gpus = envs['SLURM_STEP_GPUS'].split(',')
                            for gpu in gpus:
                                gpu_set.add(int(gpu))
                        break
                else:
                    # Could not find the env variables, slurm_adopt only fill the jobid
                    account = "error"

                with open(mem_path + 'memory.usage_in_bytes', 'r') as f_usage:
                    gauge_memory_usage.add_metric([user, account, job], int(f_usage.read()))
                with open(mem_path + 'memory.max_usage_in_bytes', 'r') as f_max:
                    gauge_memory_max.add_metric([user, account, job], int(f_max.read()))
                with open(mem_path + 'memory.limit_in_bytes', 'r') as f_limit:
                    gauge_memory_limit.add_metric([user, account, job], int(f_limit.read()))

                with open(mem_path + 'memory.stat', 'r') as f_stats:
                    for line in f_stats.readlines():
                        data = line.split()
                        if data[0] == 'total_cache':
                            gauge_memory_cache.add_metric([user, account, job], int(data[1]))
                        elif data[0] == 'total_rss':
                            gauge_memory_rss.add_metric([user, account, job], int(data[1]))
                        elif data[0] == 'total_rss_huge':
                            gauge_memory_rss_huge.add_metric([user, account, job], int(data[1]))
                        elif data[0] == 'total_mapped_file':
                            gauge_memory_mapped_file.add_metric([user, account, job], int(data[1]))
                        elif data[0] == 'total_active_file':
                            gauge_memory_active_file.add_metric([user, account, job], int(data[1]))
                        elif data[0] == 'total_inactive_file':
                            gauge_memory_inactive_file.add_metric([user, account, job], int(data[1]))
                        elif data[0] == 'total_unevictable':
                            gauge_memory_unevictable.add_metric([user, account, job], int(data[1]))

                # get the allocated cores
                with open('/sys/fs/cgroup/cpuset/slurm/uid_{}/job_{}/\
cpuset.effective_cpus'.format(uid, job), 'r') as f_cores:
                    cores = split_range(f_cores.read())
                with open('/sys/fs/cgroup/cpu,cpuacct/slurm/uid_{}/job_{}/\
cpuacct.usage_percpu'.format(uid, job), 'r') as f_usage:
                    cpu_usages = f_usage.read().split()
                    for core in cores:
                        counter_core_usage.add_metric([user, account, job, str(core)],
                                                      int(cpu_usages[core]))

                if MONITOR_GPU:
                    for gpu in gpu_set:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
                        gpu_type = pynvml.nvmlDeviceGetName(handle).decode()
                        gauge_memory_usage_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type],
                            int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
                        gauge_power_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type],
                            pynvml.nvmlDeviceGetPowerUsage(handle))
                        utils = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gauge_utilization_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type], utils.gpu)
                        gauge_memory_utilization_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type], utils.memory)
                        gauge_pcie_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type, 'TX'],
                            pynvml.nvmlDeviceGetPcieThroughput(handle, 0))
                        gauge_pcie_gpu.add_metric(
                            [user, account, job, str(gpu), gpu_type, 'RX'],
                            pynvml.nvmlDeviceGetPcieThroughput(handle, 1))

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

        if MONITOR_GPU:
            yield gauge_memory_usage_gpu
            yield gauge_power_gpu
            yield gauge_utilization_gpu
            yield gauge_memory_utilization_gpu
            yield gauge_pcie_gpu


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
    ARGS = PARSER.parse_args()

    APP = make_wsgi_app(SlurmJobCollector())
    HTTPD = make_server('', ARGS.port, APP,
                        handler_class=NoLoggingWSGIRequestHandler)
    HTTPD.serve_forever()
