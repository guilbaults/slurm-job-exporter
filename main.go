package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/NVIDIA/go-dcgm/pkg/dcgm"
	"github.com/containerd/cgroups"
	"github.com/cri-o/cpuset"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/prometheus/procfs"
)

var dcgm_enabled bool
var nvml_enable bool

type jobCollector struct {
	slurm_job_memory_usage         *prometheus.Desc
	slurm_job_memory_max           *prometheus.Desc
	slurm_job_memory_limit         *prometheus.Desc
	slurm_job_memory_cache         *prometheus.Desc
	slurm_job_memory_rss           *prometheus.Desc
	slurm_job_memory_rss_huge      *prometheus.Desc
	slurm_job_memory_mapped_file   *prometheus.Desc
	slurm_job_memory_active_file   *prometheus.Desc
	slurm_job_memory_inactive_file *prometheus.Desc
	slurm_job_memory_unevictable   *prometheus.Desc
	slurm_job_core_usage           *prometheus.Desc
	slurm_job_process_count        *prometheus.Desc
	slurm_job_threads_count        *prometheus.Desc
	slurm_job_memory_usage_gpu     *prometheus.Desc
}

func newJobCollector() *jobCollector {
	return &jobCollector{
		slurm_job_memory_usage: prometheus.NewDesc("slurm_job_memory_usage",
			"Memory used by a job",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
		slurm_job_memory_max: prometheus.NewDesc("slurm_job_memory_max",
			"Maximum memory used by a job",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
		slurm_job_memory_limit: prometheus.NewDesc("slurm_job_memory_limit",
			"Memory limit of a job",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
		slurm_job_memory_cache: prometheus.NewDesc("slurm_job_memory_cache",
			"bytes of page cache memory",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
		slurm_job_memory_rss: prometheus.NewDesc("slurm_job_memory_rss",
			"bytes of anonymous and swap cache memory (includes transparent hugepages).",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
		slurm_job_memory_rss_huge: prometheus.NewDesc("slurm_job_memory_rss_huge",
			"bytes of anonymous transparent hugepages",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
		slurm_job_memory_mapped_file: prometheus.NewDesc("slurm_job_memory_mapped_file",
			"bytes of mapped file (includes tmpfs/shmem)",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
		slurm_job_memory_active_file: prometheus.NewDesc("slurm_job_memory_active_file",
			"bytes of file-backed memory on active LRU list",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
		slurm_job_memory_inactive_file: prometheus.NewDesc("slurm_job_memory_inactive_file",
			"bytes of file-backed memory on inactive LRU list",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
		slurm_job_memory_unevictable: prometheus.NewDesc("slurm_job_memory_unevictable",
			"bytes of memory that cannot be reclaimed (mlocked etc)",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
		slurm_job_core_usage: prometheus.NewDesc("slurm_job_core_usage",
			"Cpu usage of cores allocated to a job",
			[]string{"account", "slurmjobid", "user", "core"}, nil,
		),
		slurm_job_process_count: prometheus.NewDesc("slurm_job_process_count",
			"Number of processes in a job",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
		slurm_job_threads_count: prometheus.NewDesc("slurm_job_threads_count",
			"Number of threads in a job",
			[]string{"account", "slurmjobid", "user", "state"}, nil,
		),
		slurm_job_memory_usage_gpu: prometheus.NewDesc("slurm_job_memory_usage_gpu",
			"Memory used by a job on a GPU",
			[]string{"account", "slurmjobid", "user"}, nil,
		),
	}
}

func (collector *jobCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- collector.slurm_job_memory_usage
	ch <- collector.slurm_job_memory_max
	ch <- collector.slurm_job_memory_limit
	ch <- collector.slurm_job_memory_cache
	ch <- collector.slurm_job_memory_rss
	ch <- collector.slurm_job_memory_rss_huge
	ch <- collector.slurm_job_memory_mapped_file
	ch <- collector.slurm_job_memory_active_file
	ch <- collector.slurm_job_memory_inactive_file
	ch <- collector.slurm_job_memory_unevictable
	ch <- collector.slurm_job_core_usage
}

func (collector *jobCollector) Collect(ch chan<- prometheus.Metric) {
	files, _ := filepath.Glob("/sys/fs/cgroup/memory/slurm/uid_*")
	for _, file := range files {
		uid := filepath.Base(file)[4:]
		user, _ := user.LookupId(uid)

		job_paths, _ := filepath.Glob("/sys/fs/cgroup/memory/slurm/uid_" + uid + "/job_*")
		for _, job_path := range job_paths {
			// for each job on the node
			job_id := filepath.Base(job_path)[4:]

			control, err := cgroups.Load(cgroups.V1, cgroups.StaticPath("/slurm/uid_"+uid+"/job_"+job_id))
			if err != nil {
				log.Fatal(err)
			}
			stats, err := control.Stat(cgroups.IgnoreNotExist)
			if err != nil {
				log.Fatal(err)
			}

			processes, err := control.Processes(cgroups.Devices, true)
			if err != nil {
				log.Fatal(err)
			}

			var account string = ""
			for _, process := range processes {
				pid := process.Pid

				proc, err := procfs.NewProc(pid)
				if err != nil {
					log.Fatal(err)
				}

				r_account, err := regexp.Compile("SLURM_JOB_ACCOUNT=(.+)")
				if err != nil {
					log.Fatal(err)
				}

				envs, _ := proc.Environ()
				for _, env := range envs {
					if r_account.MatchString(env) {
						account = r_account.FindStringSubmatch(env)[1]
						break
					}
				}
			}

			process_count := len(processes)
			if process_count != 0 {
				ch <- prometheus.MustNewConstMetric(collector.slurm_job_process_count, prometheus.GaugeValue,
					float64(process_count),
					account, job_id, user.Username)
			}

			ch <- prometheus.MustNewConstMetric(collector.slurm_job_memory_usage, prometheus.GaugeValue,
				float64(stats.Memory.Usage.Usage),
				account, job_id, user.Username)
			ch <- prometheus.MustNewConstMetric(collector.slurm_job_memory_max, prometheus.GaugeValue,
				float64(stats.Memory.Usage.Max),
				account, job_id, user.Username)
			ch <- prometheus.MustNewConstMetric(collector.slurm_job_memory_limit, prometheus.GaugeValue,
				float64(stats.Memory.HierarchicalMemoryLimit),
				account, job_id, user.Username)
			ch <- prometheus.MustNewConstMetric(collector.slurm_job_memory_cache, prometheus.GaugeValue,
				float64(stats.Memory.Cache),
				account, job_id, user.Username)
			ch <- prometheus.MustNewConstMetric(collector.slurm_job_memory_rss, prometheus.GaugeValue,
				float64(stats.Memory.RSS),
				account, job_id, user.Username)
			ch <- prometheus.MustNewConstMetric(collector.slurm_job_memory_rss_huge, prometheus.GaugeValue,
				float64(stats.Memory.RSSHuge),
				account, job_id, user.Username)
			ch <- prometheus.MustNewConstMetric(collector.slurm_job_memory_mapped_file, prometheus.GaugeValue,
				float64(stats.Memory.MappedFile),
				account, job_id, user.Username)
			ch <- prometheus.MustNewConstMetric(collector.slurm_job_memory_active_file, prometheus.GaugeValue,
				float64(stats.Memory.ActiveFile),
				account, job_id, user.Username)
			ch <- prometheus.MustNewConstMetric(collector.slurm_job_memory_inactive_file, prometheus.GaugeValue,
				float64(stats.Memory.InactiveFile),
				account, job_id, user.Username)
			ch <- prometheus.MustNewConstMetric(collector.slurm_job_memory_unevictable, prometheus.GaugeValue,
				float64(stats.Memory.Unevictable),
				account, job_id, user.Username)

			content, err := os.ReadFile("/sys/fs/cgroup/cpuset/slurm/uid_" + uid + "/job_" + job_id + "/cpuset.effective_cpus")
			if err != nil {
				log.Fatal(err)
			}
			cpuset_str := string(strings.TrimSuffix(string(content), "\n"))
			parsed_cpuset, err := cpuset.Parse(cpuset_str)
			if err != nil {
				log.Fatal(err)
			}
			cpuset_slice := parsed_cpuset.ToSlice()
			for i := range cpuset_slice {
				ch <- prometheus.MustNewConstMetric(collector.slurm_job_core_usage, prometheus.CounterValue,
					float64(stats.CPU.Usage.PerCPU[cpuset_slice[i]]),
					account, job_id, user.Username, fmt.Sprintf("%v", cpuset_slice[i]))
			}

			if stats.CgroupStats != nil {
				ch <- prometheus.MustNewConstMetric(collector.slurm_job_threads_count, prometheus.GaugeValue,
					float64(stats.CgroupStats.NrRunning),
					account, job_id, user.Username, "running")
				ch <- prometheus.MustNewConstMetric(collector.slurm_job_threads_count, prometheus.GaugeValue,
					float64(stats.CgroupStats.NrSleeping),
					account, job_id, user.Username, "sleeping")
				ch <- prometheus.MustNewConstMetric(collector.slurm_job_threads_count, prometheus.GaugeValue,
					float64(stats.CgroupStats.NrIoWait),
					account, job_id, user.Username, "disk-sleep")
			}

			if dcgm_enabled {
				fields := []dcgm.Short{
					dcgm.DCGM_FI_PROF_GR_ENGINE_ACTIVE,        //float64
					dcgm.DCGM_FI_PROF_SM_ACTIVE,               //float64
					dcgm.DCGM_FI_PROF_SM_OCCUPANCY,            //float64
					dcgm.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,      //float64
					dcgm.DCGM_FI_PROF_DRAM_ACTIVE,             //float64
					dcgm.DCGM_FI_PROF_PIPE_FP64_ACTIVE,        //float64
					dcgm.DCGM_FI_PROF_PIPE_FP32_ACTIVE,        //float64
					dcgm.DCGM_FI_PROF_PIPE_FP16_ACTIVE,        //float64
					dcgm.DCGM_FI_PROF_PCIE_TX_BYTES,           //int64
					dcgm.DCGM_FI_PROF_PCIE_RX_BYTES,           //int64
					dcgm.DCGM_FI_PROF_NVLINK_TX_BYTES,         //int64
					dcgm.DCGM_FI_PROF_NVLINK_RX_BYTES,         //int64
					dcgm.DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE, //int64
					dcgm.DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE, //int64
					dcgm.DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE, //int64
					dcgm.DCGM_FI_PROF_PIPE_INT_ACTIVE,         //int64
				}

				//var fieldsGroup dcgm.FieldHandle
				//dcgm.WatchFields(0, fieldsGroup, "slurm-job")
				//dcgm.UpdateAllFields()

				var values []dcgm.FieldValue_v1
				values, err = dcgm.GetLatestValuesForFields(0, fields) // do not return the correct values
				// since dcgm-exporter does not use this function, it might be a bug in the library
				if err != nil {
					log.Fatal(err)
				}
				for _, value := range values {
					switch value.FieldType {
					case dcgm.DCGM_FT_DOUBLE:
						v := value.Float64()
						log.Println("DCGM latest value float:", value.FieldId, v)
					case dcgm.DCGM_FT_INT64:
						v := value.Int64()
						log.Println("DCGM latest value int:", value.FieldId, v)
					case dcgm.DCGM_FT_STRING:
						v := value.String()
						log.Println("DCGM latest value string:", value.FieldId, v)
					default:
						log.Fatal("DCGM failed to convert value")
					}
				}
			}
		}
	}
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	r := prometheus.NewRegistry()
	job_collector := newJobCollector()
	r.MustRegister(job_collector)

	cleanup, err := dcgm.Init(dcgm.Standalone, "localhost:5555", "0")
	defer cleanup()
	if err != nil {
		log.Fatal(err)
		log.Println("DCGM disabled")
		dcgm_enabled = false
	} else {
		log.Println("DCGM initialized")
		dcgm_enabled = true
	}
	nvml_enable = false //TODO as a fallback if dcgm is not available

	handler := promhttp.HandlerFor(r, promhttp.HandlerOpts{})
	http.Handle("/metrics", handler)
	http.ListenAndServe(":9101", nil)
}
