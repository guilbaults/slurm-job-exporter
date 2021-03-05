from prometheus_client.core import REGISTRY, GaugeMetricFamily, CounterMetricFamily
from prometheus_client import start_http_server
import glob
import os
import argparse
import time
import subprocess

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

@lru_cache(maxsize=100)
def get_username(uid):
    command = ['/usr/bin/id', '--name', '--user', '{}'.format(uid)]
    return subprocess.check_output(command).strip()

def cgroup_processes(uid, job):
    procs = []
    for i in ['step_batch', 'step_extern']:
        g = '/sys/fs/cgroup/memory/slurm/uid_{}/job_{}/{}/task_*'.format(uid, job, i)
        for process_file in glob.glob(g):
            with open(process_file + '/tasks', 'r') as f:
                for proc in f.readline():
                    procs.append(proc)
    return procs

def split_range(s):
    # split a range such as "0-1,3,5,10-13"
    # to 0,1,3,5,10,11,12,13
    ranges = []
    for sub in s.split(','):
        if '-' in sub:
            r = sub.split('-')
            for i in range(int(r[0]), int(r[1]) + 1):
                ranges.append(i)
        else:
            ranges.append(int(sub))
    return ranges

class SlurmJobCollector(object):
    def collect(self):
        gauge_memory_usage = GaugeMetricFamily(
            'memory_usage', 'Memory used by a job',
            labels=['user', 'job'])
        gauge_memory_max = GaugeMetricFamily(
            'memory_max', 'Maximum memory used by a job',
            labels=['user', 'job'])
        gauge_memory_limit = GaugeMetricFamily(
            'memory_limit', 'Memory limit of a job',
            labels=['user', 'job'])
        counter_core_usage = CounterMetricFamily(
            'core_usage', 'Cpu usage of cores allocated to a job',
            labels=['user', 'job', 'core'])

        for uid_dir in glob.glob("/sys/fs/cgroup/memory/slurm/uid_*"):
            uid = uid_dir.split('/')[-1].split('_')[1]
            job_path = "/sys/fs/cgroup/memory/slurm/uid_{}/job_*".format(uid)
            for job_dir in glob.glob(job_path):
                job = job_dir.split('/')[-1].split('_')[1]
                mem_path = '/sys/fs/cgroup/memory/slurm/uid_{}/job_{}/'.format(uid, job)
                procs = cgroup_processes(uid, job)
                if len(procs) == 0:
                    continue

                # Job is alive, we can get the stats
                user = get_username(uid)
                with open(mem_path + 'memory.usage_in_bytes') as f:
                    gauge_memory_usage.add_metric([user, job], int(f.read()))
                with open(mem_path + 'memory.max_usage_in_bytes') as f:
                    gauge_memory_max.add_metric([user, job], int(f.read()))
                with open(mem_path + 'memory.limit_in_bytes') as f:
                    gauge_memory_limit.add_metric([user, job], int(f.read()))

                # get the allocated cores
                with open('/sys/fs/cgroup/cpuset/slurm/uid_{}/job_{}/cpuset.effective_cpus'.format(uid, job)) as f:
                    cores = split_range(f.read())
                with open('/sys/fs/cgroup/cpu,cpuacct/slurm/uid_{}/job_{}/cpuacct.usage_percpu'.format(uid, job)) as f:
                    cpu_usages = f.read().split()
                    for core in cores:
                        counter_core_usage.add_metric([user, job, str(core)], int(cpu_usages[core]))

        yield gauge_memory_usage
        yield gauge_memory_max
        yield gauge_memory_limit
        yield counter_core_usage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TODO')
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Collector http port, default is 8080')
    args = parser.parse_args()

    start_http_server(args.port)
    REGISTRY.register(SlurmJobCollector())
    while True:
        time.sleep(60)
