import numpy as np

import ros2_numpy
from ros2_numpy.point_cloud2 import merge_rgb_fields
from sensor_msgs.msg import PointCloud2


def pc_np_to_pc_msg(pc, names, frame_id):
    """

    Args:
        pc: [M, N] array where M is probably either 3 or 6
        names: strings of comma separated names of the fields in pc, e.g. 'x,y,z' or 'x,y,z,r,g,b'
        frame_id: string

    Returns:
        PointCloud2 message

    """
    pc_rec = np.rec.fromarrays(pc, names=names)
    if 'r' in names:
        pc_rec = merge_rgb_fields(pc_rec)
    pc_msg = ros2_numpy.msgify(PointCloud2, pc_rec, frame_id=frame_id)
    return pc_msg
