Name:	  slurm-job-exporter
Version:  0.3.2
Release:  1%{?dist}
Summary:  Prometheus exporter for stats in slurm accounting cgroups

License:  Apache License 2.0
URL:      https://github.com/guilbaults/slurm-job-exporter
Source0:  https://github.com/guilbaults/%{name}/archive/refs/tags/v%{version}.tar.gz

BuildArch:      noarch
BuildRequires:	systemd
Requires:       python3
Requires:       python36-psutil

%description
Prometheus exporter for the stats in the cgroup accounting with slurm. This will also collect stats of a job using NVIDIA GPUs.

%prep
%setup -q

%build

%install
mkdir -p %{buildroot}/%{_bindir}
mkdir -p %{buildroot}/%{_unitdir}

sed -i -e '1i#!/usr/bin/python3' slurm-job-exporter.py
install -m 0755 %{name}.py %{buildroot}/%{_bindir}/%{name}
install -m 0744 get_gpus.sh %{buildroot}/%{_bindir}/get_gpus.sh
install -m 0644 slurm-job-exporter.service %{buildroot}/%{_unitdir}/slurm-job-exporter.service

%clean
rm -rf $RPM_BUILD_ROOT

%files
%{_bindir}/%{name}
%{_bindir}/get_gpus.sh
%{_unitdir}/slurm-job-exporter.service

%changelog
* Fri Nov 24 2023 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.3.2-1
- Fixing support for GPUs without FP64
* Tue Nov 23 2023 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.3.1-1
- Adding support for GPUs without FP64
* Tue Sep 25 2023 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.3.0-1
- Removing the dependency on cgexec
* Tue Aug  1 2023 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.2.0-1
- Adding counter_process_usage to measure and name the processes inside the cgroups
* Fri Jul 14 2023 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.1.0-1
- Adding support for MIG devices
* Tue Apr 11 2023 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.13-1
- Fixing DCGM bandwidth gauges and adding --dcgm-update-interval parameter
* Mon Nov 29 2022 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.12-1
- slurm_job_threads_count by thread status, can now distinguish sleeping, running, io-wait ...
* Mon Nov 28 2022 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.11-1
- Measure the qty of processes and threads
* Mon Oct 31 2022 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.10-1
- Adding support of GPU-less job on a GPU node
* Tue Oct 18 2022 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.9-1
- Power stats is now optional since its not visible in NVIDIA-GRID
* Fri Jul 15 2022 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.8-1
- Handle utf8 characters in job's environment
* Tue Jul  5 2022 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.7-1
- Collecting GPU stats with Nvidia DCGM and keeping NVML as a fallback
* Fri Aug 13 2021 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.6-1
- Collecting more memory stats from each job cgroup
* Wed Apr 28 2021 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.5-1
- Adding a workaround to prevent a crash when the other nodes of a multinodes job is using slurm_adopt
* Wed Apr 14 2021 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.4-1
- Grabbing slurm account and using a better way to match GPU to a job
* Wed Apr  7 2021 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.3-1
- Replacing job by slurmjobid since it conflit with a internal field in Prometheus
* Wed Mar  10 2021 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.2-1
- Fixing a collecting bug, removing the sleep() and changing the metric prefixes to slurm_job
* Mon Mar  8 2021 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.1-1
- Initial release
