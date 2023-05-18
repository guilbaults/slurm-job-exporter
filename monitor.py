import glob
import argparse
import subprocess
import re
import sys
import psutil
from functools import lru_cache
from wsgiref.simple_server import make_server, WSGIRequestHandler
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

# Load DCGM bindings
sys.path.insert(0, '/usr/local/dcgm/bindings/python3/')

try:
    import pydcgm
    import dcgm_fields
    import dcgm_structs
    import dcgm_agent
    import dcgm_field_helpers
except ImportError:
    pydcgm = None


def GetGroupIdByName(self, name):
    for group_id in self.GetAllGroupIds():
        groupInfo = dcgm_agent.dcgmGroupGetInfo(self._dcgmHandle.handle, group_id)
        if groupInfo.groupName == name:
            return group_id

    return None


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
            with open(process_file + '/cgroup.procs', 'r') as stats:
                for proc in stats.readlines():
                    # check if process is not running as root
                    # a long sleep running as root can be found in step_extern
                    try:
                        ps = psutil.Process(int(proc))
                        if ps.username() != 'root':
                            procs.append(int(proc))
                    except psutil.NoSuchProcess:
                        pass
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
    try:
        with open('/proc/{}/environ'.format(pid), 'r', encoding='utf-8') as env_f:
            for env in env_f.read().split('\000'):
                r_env = re.match(r'(.*)=(.*)', env)
                if r_env:
                    environments[r_env.group(1)] = r_env.group(2)
    except FileNotFoundError:
        raise ValueError('Process {} environment does not exist'.format(pid))
    return environments


class SlurmJobCollector(object):
    """
    Used by a WSGI application to collect and return stats about currently
    running slurm jobs on a node. This is using the stats from the cgroups
    created by Slurm.
    """

    def __init__(self, influx_url, influx_org, influx_token):
        self.client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        if pydcgm is not None:
            self.GPU_LABEL_MAP = {
                dcgm_fields.DCGM_FI_DEV_FB_USED: "memory_usage",
                dcgm_fields.DCGM_FI_DEV_POWER_USAGE: "power",
                dcgm_fields.DCGM_FI_PROF_SM_ACTIVE: "utilization",
                dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY: "sm_occupancy",
                dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE: "tensor",
                dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE: "memory_utilization",
                dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE: "fp64",
                dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE: "fp32",
                dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE: "fp16",
                dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES: "pcie_tx",
                dcgm_fields.DCGM_FI_PROF_PCIE_RX_BYTES: "pcie_rx",
                dcgm_fields.DCGM_FI_PROF_NVLINK_TX_BYTES: "nvlink_tx",
                dcgm_fields.DCGM_FI_PROF_NVLINK_RX_BYTES: "nvlink_rx",
                }

    def collect(self):
        """
        Run a collection cycle and update exported stats
        """
        if pydcgm is not None:
            handle = pydcgm.DcgmHandle(None, 'localhost')
            system = handle.GetSystem()
        
        
        points = []
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
                for proc in procs:
                    try:
                        envs = get_env(proc)
                    except ValueError:
                        # Process does not have an environment, its probably gone
                        continue
                    if 'SLURM_JOB_ACCOUNT' in envs:
                        account = envs['SLURM_JOB_ACCOUNT']

                else:
                    # Could not find the env variables, slurm_adopt only fill the jobid
                    account = "error"

                def basic_tag(p):
                    # Maybe tag with cluster and/or node
                    p.tag("user", user)
                    p.tag("account", account)
                    p.tag("slurmjobid", job)

                p = influxdb_client.Point("slurm_job")
                basic_tag(p)
                points.append(p)
                with open(mem_path + 'memory.usage_in_bytes', 'r') as f_usage:
                    p.field("memory_usage", int(f_usage.read()))

                with open(mem_path + 'memory.max_usage_in_bytes', 'r') as f_max:
                    p.field("memory_max", int(f_max.read()))

                with open(mem_path + 'memory.limit_in_bytes', 'r') as f_limit:
                    p.field("memory_limit", int(f_limit.read()))

                with open(mem_path + 'memory.stat', 'r') as f_stats:
                    for line in f_stats.readlines():
                        data = line.split()
                        if data[0] == 'total_cache':
                            p.field("memory_cache", int(data[1]))
                        elif data[0] == 'total_rss':
                            p.field("memory_rss", int(data[1]))
                        elif data[0] == 'total_rss_huge':
                            p.field("memory_rss_huge", int(data[1]))
                        elif data[0] == 'total_mapped_file':
                            p.field("memory_mapped_file", int(data[1]))
                        elif data[0] == 'total_active_file':
                            p.field("memory_active_file", int(data[1]))
                        elif data[0] == 'total_inactive_file':
                            p.field("memory_inactive_file", int(data[1]))
                        elif data[0] == 'total_unevictable':
                            p.field("memory_unevictable", int(data[1]))

                # get the allocated cores
                with open('/sys/fs/cgroup/cpuset/slurm/uid_{}/job_{}/cpuset.effective_cpus'.format(uid, job), 'r') as f_cores:
                    cores = split_range(f_cores.read())
                with open('/sys/fs/cgroup/cpu,cpuacct/slurm/uid_{}/job_{}/cpuacct.usage_percpu'.format(uid, job), 'r') as f_usage:
                    cpu_usages = f_usage.read().split()
                    for core in cores:
                        pc = influxdb_client.Point("slurm_job_percore")
                        basic_tag(pc)
                        pc.tag("core", str(core))
                        pc.field("cpu_usage", int(cpu_usages[core]))
                        points.append(pc)
                    # Add an overall summary
                    p.field("cpu_usage", sum(cpu_usages)/len(cpu_usages))

                processes = 0
                tasks_state = {}
                for proc in procs:
                    try:
                        p = psutil.Process(proc)
                    except psutil.NoSuchProcess:
                        continue
                    cmdline = p.cmdline()
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
                    pt = influxdb_client.Point("slurm_job_thread_perstate")
                    basic_tag(pt)
                    pt.tag("state", status)
                    pt.field("count", tasks_state[status))
                    points.append(pt)
                p.field("process_count", processes)

                # This is skipped if we can't import the DCGM bindings
                if pydcgm is not None:
                    fg_id = system.GetFieldGroupIdByName(f"{job_id}-fg")
                    g_id = GetGroupIdByName(job_id)
                    dcgm_data = dcgm_field_helpers.DcgmFieldValueCollection(handle.handle, g_id)
                    # This will fill in dcgm_data with callbacks
                    dcgm_data.GetLatestValues_v2(fg_id)

                    for gpu, data in dcgm_data.items():
                        pg = influxdb_client.Point("slurm_job_gpudata")
                        basic_tag(pg)
                        pg.tag("gpu", gpu)

                        gpu_type = get_value(data, dcgm_fields.DCGM_FI_DEV_NAME)
                        pg.tag("gpu_type", gpu_type)

                        for field, cell in data.items():
                            v = cell.values[0]
                            if v.isBlank:
                                continue
                            pg.field(self.GPU_LABEL_MAP[field], v.value)
                        points.append(pg)
        # XXX: May have to do something extra to enable batching
        self.write_api.write(points)


if __name__ == '__main__':
    # parse config file
    config = ...
    collector = SlurmJobCollector(influx_url=config.influx_url, influx_token=config.influx_token, influx_org=config.influx_org)
    while True:
        sleep(config.interval)
        collector.collect()
