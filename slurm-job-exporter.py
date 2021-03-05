from prometheus_client.core import REGISTRY, GaugeMetricFamily
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


class SlurmJobCollector(object):
    def collect(self):
        gauge_memory_usage = GaugeMetricFamily(
            'memory_usage', 'Memory used by a job',
            labels=['user', 'job'])

        for uid_dir in glob.glob("/sys/fs/cgroup/memory/slurm/uid_*"):
            uid = uid_dir.split('/')[-1].split('_')[1]
            job_path = "/sys/fs/cgroup/memory/slurm/uid_{}/job_*".format(uid)
            for job_dir in glob.glob(job_path):
                job = job_dir.split('/')[-1].split('_')[1]
                path = '/sys/fs/cgroup/memory/slurm/uid_{}/job_{}/'.format(uid, job)
                procs = cgroup_processes(uid, job)
                if len(procs) == 0:
                    continue

                # Job is alive, we can get the stats
                user = get_username(uid)
                with open(path + 'memory.usage_in_bytes') as f:
                    memory_usage = int(f.read())
                    gauge_memory_usage.add_metric([user, job], memory_usage)
        yield gauge_memory_usage

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
