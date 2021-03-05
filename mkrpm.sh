#!/bin/bash
spectool -g -R slurm-job-exporter-el7.spec
rpmbuild --define "dist .el7" -ba slurm-job-exporter-el7.spec
