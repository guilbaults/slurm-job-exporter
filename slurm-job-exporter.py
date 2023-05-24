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
    import dcgm_agent
    import dcgm_field_helpers


    GPU_LABEL_MAP = {
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
        dcgm_fields.DCGM_FI_DEV_NVML_INDEX,
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
                sleep(config.interval)
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
                metric = GaugeMetricFamily(f"{point.name}_{name}", "", labels=labels)
                metric.add_metric(list(point.tags.values()), value)
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

    def field(name, value):
        self.fields[name] = value
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
            self.group = pydcgm.DcgmGroup(self.handle, groupName="slurm-job-exporter", groupType=dcgm_structs.DCGM_GROUP_ALL_INSTANCES)
            self.is_mig = True
            if len(self.group.GetEntities) == 0:
                self.is_mig = False
                self.group = pydcgm.DcgmGroup(self.handle, groupName="slurm-job-exporter", groupType=dcgm_structs.DCGM_GROUP_ALL_GPUS)
            self.field_group = pydcgm.DcgmFieldGroup(self.handle, name="slurm-job-exporter-fg", fieldIds=FIELDS_MIG if self.is_mig else FIELDS_GPU)
            self.group.samples.WatchFields(self.field_group, config.dcgm_update_interval*1000*1000, config.dcgm_update_interval*2.0, 0)
            self.group.UpdateAllFields(1)

    def collect(self):
        """
        Run a collection cycle and update exported stats
        """
        if pydcgm is not None:
            gpu_data = self.group.GetLatest_v2(self.field_group)[dcgm_fields.DCGM_FE_GPU_I if self.is_mig else dcgm_fields.DCGM_FE_GPU]

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

                account = "error"
                visible_gpus = None
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
                    if self.is_mig and 'CUDA_VISIBLE_DEVICES' in envs:
                        visible_gpus = envs['CUDA_VISIBLE_DEVICES']
                    if not self.is_mig
                        if 'SLURM_JOB_GPUS' in envs:
                            visible_gpus = envs['SLURM_JOB_GPUS']
                        elif 'SLURM_STEP_GPUS' in envs:
                            visible_gpus = envs['SLURM_STEP_GPUS']


                def basic_tag(p):
                    # Maybe tag with cluster and/or node
                    p.tag("user", user)
                    p.tag("account", account)
                    p.tag("slurmjobid", job)

                p = Point("slurm_job")
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
                        pc = Point("slurm_job_percore")
                        basic_tag(pc)
                        pc.tag("core", str(core))
                        pc.field("cpu_usage", int(cpu_usages[core]))
                        points.append(pc)

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
                    pt = Point("slurm_job_thread_perstate")
                    basic_tag(pt)
                    pt.tag("state", status)
                    pt.field("count", tasks_state[status])
                    points.append(pt)
                p.field("process_count", processes)

                # This is skipped if we can't import the DCGM bindings
                # or there are no GPUs in use by the job
                if pydcgm is not None and visible_gpus is not None:
                    gpus = visible_gpus.split(',')
                    for gpu in gpus:
                        for gdata in dcgm_data:
                            if self.is_mig:
                                uuid = data.pop(dcgm_fields.DCGM_FI_DEV_UUID).values[0].value
                                if uuid != gpu:
                                    continue
                            else:
                                index = data.pop(dcgm_fields.DCGM_FI_DEV_NVML_INDEX).values[0].value
                                if index != gpu:
                                    continue

                            pg = Point("slurm_job_gpudata")
                            basic_tag(pg)
                            pg.tag("gpu", gpu)
                            pg.tag("gpu_type", data.pop(dcgm_fields.DCGM_FI_DEV_NAME).values[0].value)

                            for field, cell in data.items():
                                v = cell.values[0]
                                if v.isBlank:
                                    continue
                                pg.field(self.GPU_LABEL_MAP[field], v.value)
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
