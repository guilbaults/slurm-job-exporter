import sys

# Load DCGM bindings
sys.path.insert(0, '/usr/local/dcgm/bindings/python3/')

try:
    import pydcgm
    import dcgm_fields
    import dcgm_structs
    import dcgm_agent
except ImportError:
    pydcgm = None


def add_mig(handle, uuid, g_id):
    # Makes a group with the mig device identified by `uuid`
    g = pydcgm.DcgmGroup(handle, groupName=f"find-{uuid}", groupType=dcgm_structs.DCGM_GROUP_DEFAULT_INSTANCES)
    fg = pydcgm.DcgmFieldGroup(handle, name=f"find-{uuid}-fg", fieldIds=[dcgm_fields.DCGM_FI_DEV_UUID])
    g.samples.WatchFields(fg, 10*1000*1000, 10.0, 0)
    g.UpdateAllFields(1)
    data = g.samples.GetLatest_v2(fg)[dcgm_fields.DCGM_FE_GPU_I]
    for mig_id, datum in data.entries():
        f = datum[dcgm_fields.DCGM_FI_DEV_UUID].values[0]
        assert not f.isBlank
        if f.value == uuid:
            break
   g.samples.UnwatchFields(fg)
   g.Delete()
   ret = dcgm_agent.dcgmGroupAddEntity(handle._handle, g_id, dcgm_fields.DCGM_FE_GPU_I, mig_id)
   dcgm_structs._dcgmCheckReturn(ret)


def add_gpus(handle, devices, g_id):
    # XXX: make sure that the gpu ids we recieve match what we expect
    # maybe we can get more info about our GPUs from slurm?
    # Ideally we would use UUIDs like for MIG above
    for dev in devices:
        ret = dcgm_agent.dcgmGroupAddDevice(handle._handle, g_id, int(dev))
        dcgm_structs._dcgmCheckReturn(ret)


def monitor_devices(devices, job_id):
    if len(devices) < 0:
        print("no GPUS for job, not monitoring")
        return

    if pydcgm is None:
        print("Could not load DCGM bindings, not monitoring")
        return

    handle = pydcgm.DcgmHandle(None, 'localhost')
    g_id = dcgm_agent.dcgmGroupCreate(handle._handle, group_type=dcgm_structs.DCGM_GROUP_EMPTY, group_name=job_id)

    if len(devices) == 1 and dev.startswith("MIG-"):
        group = add_mig(handle, devices[0], g_id)
        # TODO: Make sure what fields are working for MIG
        field_ids = [
            dcgm_fields.DCGM_FI_DEV_NAME,
            dcgm_fields.DCGM_FI_DEV_FB_USED,
            dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY,
            dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE,
            ]
    else:  # we assume no MIG here
        group = add_gpus(handle, devices, g_id)
        field_ids = [
            dcgm_fields.DCGM_FI_DEV_NAME,
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
    fg_id = dcgm_agent.dcgmFieldGroupCreate(handle._handle, fieldIds, f"{job_id}-fg")
    updateFreq = 10 * 1000 * 1000
    maxKeepAge = 120.0
    maxKeepSamples = 0
    ret = dcgm_agent.dcgmWatchFields(handle._handle, g_id, fg_id, updateFreq, maxKeepAge, maxKeepSamples)
    dcgm_structs._dcgmCheckReturn(ret)


# get the actual job id (including step?)
JOB_ID = "test"

# MIG-<uuid>
# numbers -> full GPUS
DEVICES = ["MIG-<uuid>"] # or ["0", "1", ...]

#monitor_devices(DEVICES, JOB_ID)
