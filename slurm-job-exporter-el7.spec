Name:	  slurm-job-exporter
Version:  0.0.3
Release:  1%{?dist}
Summary:  Prometheus exporter for stats in slurm accounting cgroups

License:  Apache License 2.0
URL:      https://github.com/guilbaults/slurm-job-exporter
Source0:  https://github.com/guilbaults%{name}/%{name}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:	systemd
Requires:       python3

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
install -m 0644 slurm-job-exporter.service %{buildroot}/%{_unitdir}/slurm-job-exporter.service

%clean
rm -rf $RPM_BUILD_ROOT

%files
%{_bindir}/%{name}
%{_unitdir}/slurm-job-exporter.service

%changelog
* Wed Apr  7 2021 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.3-1
- Replacing job by slurmjobid since it conflit with a internal field in Prometheus
* Wed Mar  10 2021 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.2-1
- Fixing a collecting bug, removing the sleep() and changing the metric prefixes to slurm_job
* Mon Mar  8 2021 Simon Guilbault <simon.guilbault@calculquebec.ca> 0.0.1-1
- Initial release
