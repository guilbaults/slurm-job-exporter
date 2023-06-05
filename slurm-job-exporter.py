import glob
import argparse
import subprocess
import re
import sys
import psutil
from functools import lru_cache

# Load DCGM bindings
sys.path.insert(0, '/usr/local/dcgm/bindings/python3/')

try:
    import pydcgm
    import dcgm_fields
    import dcgm_structs

    def percent(v):
        return v.value * 100

    GPU_LABEL_MAP = {
        dcgm_fields.DCGM_FI_DEV_FB_USED: ("memory_usage_gpu", 'Memory used by a job on a GPU', lambda v: v.value * 1024 * 1024),
        dcgm_fields.DCGM_FI_DEV_POWER_USAGE: ("power_gpu", 'Power used by a job on a GPU in mW', lambda v: v.value * 1000),
        dcgm_fields.DCGM_FI_PROF_SM_ACTIVE: ("utilization_gpu", 'Percent of time over the past sample period during which one or more kernels was executing on the GPU.', percent),
        dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY: ("sm_occupancy_gpu", 'The ratio of number of warps resident on an SM. (number of resident as a ratio of the theoretical maximum number of warps per elapsed cycle)', percent),
        dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE: ("tensor_gpu", 'The ratio of cycles the tensor (HMMA) pipe is active (off the peak sustained elapsed cycles)', percent),
        dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE: ("utilization_gpu_memory", 'Percent of time over the past sample period during which global (device) memory was being read or written.', percent),
        dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE: ("fp64_gpu", 'Ratio of cycles the fp64 pipe is active', percent),
        dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE: ("fp32_gpu", 'Ratio of cycles the fp32 pipe is active', percent),
        dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE: ("fp16_gpu", 'Ratio of cycles the fp16 pipe is active', percent),
        dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES: ("pcie_tx_gpu", 'PCIe tx bytes per second'),
        dcgm_fields.DCGM_FI_PROF_PCIE_RX_BYTES: ("pcie_rx_gpu", 'PCIe rx bytes per second'),
        dcgm_fields.DCGM_FI_PROF_NVLINK_TX_BYTES: ("nvlink_tx_gpu", 'Nvlink tx bytes per second'),
        dcgm_fields.DCGM_FI_PROF_NVLINK_RX_BYTES: ("nvlink_rx_gpu", 'Nvlink rx bytes per second')
        }
    FIELDS_MIG = [
        dcgm_fields.DCGM_FI_DEV_NAME,
        dcgm_fields.DCGM_FI_DEV_UUID,
        dcgm_fields.DCGM_FI_DEV_FB_USED,
        dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY,
        dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE,
        ]
    FIELDS_GPU = [
        dcgm_fields.DCGM_FI_DEV_NAME,
        dcgm_fields.DCGM_FI_DEV_UUID,
        dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
        dcgm_fields.DCGM_FI_DEV_FB_USED,
        dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY,
        dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES,
        dcgm_fields.DCGM_FI_PROF_PCIE_RX_BYTES,
        dcgm_fields.DCGM_FI_PROF_NVLINK_TX_BYTES,
        dcgm_fields.DCGM_FI_PROF_NVLINK_RX_BYTES,
        ]

except ImportError:
    pydcgm = None


try:
    import influxdb_client
    import time
    from influxdb_client.client.write_api import SYNCHRONOUS

    class InfluxDBReporter:
        def __init__(self, config, collector):
            self.client = influxdb_client.InfluxDBClient(url=config.influx_url, token=config.influx_token, org=config.influx_org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.config = config
            self.collector = collector

        def _map_point(self, point):
            p = influxdb_client.Point(point.name)
            for k, v in point.tags.items():
                p.tag(k, v)
            for k, v in point.fields.items():
                p.field(k, v)
            return p

        def run(self):
            while True:
                time.sleep(config.interval)
                influx_points = map(self._map_point, self.collector.collect())
                # XXX: May have to do something extra to enable batching
                self.write_api.write(influx_points)

except ImportError:
    InfluxDBReporter = None

try:
    from wsgiref.simple_server import make_server, WSGIRequestHandler
    from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
    from prometheus_client import make_wsgi_app

    class NoLoggingWSGIRequestHandler(WSGIRequestHandler):
        """
        Class to remove logging of WSGI
        """
        def log_message(self, format, *args):
            pass

    class PrometheusCollector:
        def __init__(self, collector):
            self.collector = collector

        def _point_to_metrics(self, point):
            labels = list(point.tags.keys())
            for name, value in point.fields.items():
                val, field_type, description = value
                if field_type == 'gauge':
                    mclass = GaugeMetricFamily
                elif field_type == 'counter':
                    mclass = CounterMetricFamily
                else:
                    raise ValueError("unknown metric type")
                metric = mclass(f"{point.name}_{name}", description, labels=labels)
                metric.add_metric(list(point.tags.values()), value[0])
                yield metric

        def collect(self):
            points = self.collector.collect()
            for p in points:
                yield from self._point_to_metrics(p)

    class PrometheusReporter:
        def __init__(self, config, collector):
            self.config = config
            self.collector = PrometheusCollector(collector)

        def run(self):
            app = make_wsgi_app(self.collector)
            httpd = make_server('', config.port, app, handler_class=NoLoggingWSGIRequestHandler)
            httpd.serve_forever()


except ImportError:
    PrometheusReporter = None


class DebugReporter:
    def __init__(self, config, collector):
        self.collector = collector

    def _print_point(self, point):
        print(f"METRIC: {point.name}")
        print("TAGS:")
        for k, v in point.tags.items():
            print(f"{k}={v}")
        print("FIELDS:")
        for k, v in point.fields.items():
            print(f"{k}={v}")

    def run(self):
        for point in self.collector.collect():
            self._print_point(point)


class Point:
    def __init__(self, name):
        self.name = name
        self.tags = dict()
        self.fields = dict()

    def tag(self, name, value):
        self.tags[name] = value
        return self

    def field(self, name, value, *, field_type="gauge", description=""):
        self.fields[name] = (value, field_type, description)
        return self


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


GPU_UUID_RE = re.compile('(GPU|MIG)-[a-z0-9]{8}-([a-z0-9]{4}-){3}[a-z0-9]{12}')


def cgroup_gpus(uid, job, is_mig):
    try:
        command = ["cgexec", "-g", f"devices:slurm/uid_{uid}/job_{job}/", "nvidia-smi", "-L"]
        res = subprocess.check_output(command).strip().decode()
    except FileNotFoundError:
        # This is most likely because cgexec or nvidia-smi are not on the machine
        return []
    gpus = []
    for line in res.split('\n'):
        if is_mig and "MIG" not in line:
            continue
        m = GPU_UUID_RE.search(line)
        if m is not None:
            gpus.append(m.group(0))
    return gpus


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


class SlurmJobCollector:
    """Collect and return stats about currently running slurm jobs on a node.

    This is using the stats from the cgroups created by Slurm.
    """

    def __init__(self, config):
        self.is_mig = False
        if pydcgm is not None:
            self.handle = pydcgm.DcgmHandle(None, 'localhost')
            self.group = pydcgm.DcgmGroup(self.handle, groupName="slurm-job-exporter", groupType=dcgm_structs.DCGM_GROUP_DEFAULT_INSTANCES)
            self.is_mig = True
            if len(self.group.GetEntities()) == 0:
                self.group.Delete()
                self.is_mig = False
                self.group = pydcgm.DcgmGroup(self.handle, groupName="slurm-job-exporter", groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
            self.field_group = pydcgm.DcgmFieldGroup(self.handle, name="slurm-job-exporter-fg", fieldIds=FIELDS_MIG if self.is_mig else FIELDS_GPU)
            self.group.samples.WatchFields(self.field_group, config.dcgm_update_interval*1000*1000, config.dcgm_update_interval*2.0, 0)
            self.handle.GetSystem().UpdateAllFields(True)

    def collect(self):
        """
        Run a collection cycle and update exported stats
        """
        if pydcgm is not None:
            gpu_data = self.group.samples.GetLatest_v2(self.field_group).values[dcgm_fields.DCGM_FE_GPU_I if self.is_mig else dcgm_fields.DCGM_FE_GPU]

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

                visible_gpus = cgroup_gpus(uid, job, self.is_mig)

                account = "error"
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

                def basic_tag(p):
                    p.tag("user", user)
                    p.tag("account", account)
                    p.tag("slurmjobid", job)

                p = Point("slurm_job")
                basic_tag(p)
                points.append(p)
                with open(mem_path + 'memory.usage_in_bytes', 'r') as f_usage:
                    p.field("memory_usage", int(f_usage.read()), description='Memory used by a job')

                with open(mem_path + 'memory.max_usage_in_bytes', 'r') as f_max:
                    p.field("memory_max", int(f_max.read()), description='Maximum memory used by a job')

                with open(mem_path + 'memory.limit_in_bytes', 'r') as f_limit:
                    p.field("memory_limit", int(f_limit.read()), description='Memory limit of a job')

                with open(mem_path + 'memory.stat', 'r') as f_stats:
                    for line in f_stats.readlines():
                        data = line.split()
                        if data[0] == 'total_cache':
                            p.field("memory_cache", int(data[1]), description='bytes of page cache memory')
                        elif data[0] == 'total_rss':
                            p.field("memory_rss", int(data[1]), description='bytes of anonymous and swap cache memory (includes transparent hugepages).')
                        elif data[0] == 'total_rss_huge':
                            p.field("memory_rss_huge", int(data[1]), description='bytes of anonymous transparent hugepages')
                        elif data[0] == 'total_mapped_file':
                            p.field("memory_mapped_file", int(data[1]), description='bytes of mapped file (includes tmpfs/shmem)')
                        elif data[0] == 'total_active_file':
                            p.field("memory_active_file", int(data[1]), description='bytes of file-backed memory on active LRU list')
                        elif data[0] == 'total_inactive_file':
                            p.field("memory_inactive_file", int(data[1]), description='bytes of file-backed memory on inactive LRU list')
                        elif data[0] == 'total_unevictable':
                            p.field("memory_unevictable", int(data[1]), description='bytes of memory that cannot be reclaimed (mlocked etc)')

                # get the allocated cores
                with open('/sys/fs/cgroup/cpuset/slurm/uid_{}/job_{}/cpuset.effective_cpus'.format(uid, job), 'r') as f_cores:
                    cores = split_range(f_cores.read())
                with open('/sys/fs/cgroup/cpu,cpuacct/slurm/uid_{}/job_{}/cpuacct.usage_percpu'.format(uid, job), 'r') as f_usage:
                    cpu_usages = f_usage.read().split()
                    for core in cores:
                        pc = Point("slurm_job_core")
                        basic_tag(pc)
                        pc.tag("core", str(core))
                        pc.field("usage", int(cpu_usages[core]), field_type="counter", description='Cpu usage of cores allocated to a job')
                        points.append(pc)

                processes = 0
                tasks_state = {}
                for proc in procs:
                    try:
                        pr = psutil.Process(proc)
                    except psutil.NoSuchProcess:
                        continue
                    cmdline = pr.cmdline()
                    if len(cmdline) == 0:
                        # sometimes the cmdline is empty, we don't want to count it
                        continue
                    if cmdline[0] == '/bin/bash':
                        if len(cmdline) > 1:
                            if '/var/spool' in cmdline[1] and 'slurm_script' in cmdline[1]:
                                # This is the bash script of the job, we don't want to count it
                                continue
                    processes += 1

                    for t in pr.threads():
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
                    pt = Point("slurm_job_threads")
                    basic_tag(pt)
                    pt.tag("state", status)
                    pt.field("count", tasks_state[status], description='Number of threads in a job')
                    points.append(pt)
                p.field("process_count", processes, description='Number of processes in a job')

                # This is skipped if we can't import the DCGM bindings
                # or there are no GPUs in use by the job
                if pydcgm is not None and visible_gpus is not None:
                    for gpu in visible_gpus:
                        for gdata in gpu_data.values():
                            uuid = gdata[dcgm_fields.DCGM_FI_DEV_UUID].values[0].value
                            if uuid != gpu:
                                continue

                            pg = Point("slurm_job")
                            basic_tag(pg)
                            pg.tag("gpu", gpu)
                            pg.tag("gpu_type", gdata[dcgm_fields.DCGM_FI_DEV_NAME].values[0].value)

                            for field, cell in gdata.items():
                                if field not in GPU_LABEL_MAP:
                                    continue
                                v = cell.values[0]
                                if v.isBlank:
                                    continue
                                fname, descr, *fn = GPU_LABEL_MAP[field]
                                if fn:
                                    f, = fn
                                    v = f(v)
                                pg.field(fname, v, description=descr)
                            points.append(pg)
                            break
                        else:
                            print(f"WARNING: could not find gpu data for {gpu} (MIG: {self.is_mig})")
        return points


REPORTER_MAP = {
    'debug': DebugReporter,
    'prometheus': PrometheusReporter,
    'influxdb': InfluxDBReporter,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slurm job stats collector')
    parser.add_argument(
        '--port', type=int, default=9798, help='Prometheus collector http port')
    parser.add_argument(
        '--dcgm-update-interval', type=int, default=10, help='DCGM update interval, in seconds')
    parser.add_argument(
        '--reporter-type', choices=REPORTER_MAP.keys(), help='Which system to report stats to', required=True)
    # TODO: add config options for influx
    # Will probably need a config file since the options are supposed to be secret
    config = parser.parse_args()
    collector = SlurmJobCollector(config)
    reporter_cls = REPORTER_MAP[config.reporter_type]
    if reporter_cls is None:
        print("ERROR: choosen reporter is not available")
        sys.exit(1)
    reporter = reporter_cls(config, collector)
    reporter.run()
