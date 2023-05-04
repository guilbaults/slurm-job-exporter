import sys

# Load DCGM bindings
sys.path.insert(0, '/usr/local/dcgm/bindings/python3/')

import pydcgm
import dcgm_fields
import dcgm_structs


def GetGroupIdByName(self, name):
    for group_id in self.GetAllGroupIds():
        groupInfo = dcgm_agent.dcgmGroupGetInfo(self._dcgmHandle.handle, group_id)
        if groupInfo.groupName == name:
            return group_id

    return None


def stop_monitor_devices(job_id):
    handle = pydcgm.DcgmHandle(None, 'localhost')
    system = handle.GetSystem()
    fg_id = system.GetFieldGroupIdByName(f"{job_id}-fg")
    # This assumes there where no GPUs for this job
    if fg_id is None:
        return

    g_id = GetGroupIdByName(job_id)
    if g_id is None: 
        dcgm_agent.dcgmFieldGroupDestroy(handle._handle, fg_id)
        return

    dcgm_agent.dcgmUnwatchFields(handle._handle, g_id, fg_id)
    dcgm_agent.dcgmGroupDestroy(handle._handle, g_id)
    dcgm_agent.dcgmFieldGroupDestroy(handle._handle, fg_id)
    
