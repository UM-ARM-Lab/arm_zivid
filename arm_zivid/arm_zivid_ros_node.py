import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import zivid
import ros2_numpy
from arm_zivid.pc_np_to_pc_msg import pc_np_to_pc_msg
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2

from time import perf_counter
import time

CAMERA_FRAME = 'zivid_optical_frame'
MASK_THRESHOLD = 0.25


class ZividNode(Node):

    def __init__(self, settings_yml: Optional[Path] = None):
        super().__init__('zivid_node')
        if settings_yml is not None:
            self.settings = zivid.Settings.load(settings_yml)
        else:
            self.settings = zivid.Settings()

        self.pc_pub = self.create_publisher(PointCloud2, 'pc', 10)
        self.rgb_pub = self.create_publisher(Image, 'rgb', 10)
        self.depth_pub = self.create_publisher(Image, 'depth', 10)

    def run(self, camera):
        last_t = perf_counter()
        while rclpy.ok():
            with camera.capture(self.settings) as frame:
                a = time.perf_counter()
                point_cloud = frame.point_cloud()
                xyz_mm = point_cloud.copy_data("xyz")
                rgba = point_cloud.copy_data("rgba")

                xyz = xyz_mm / 1000.0
                rgb = rgba[:, :, :3]
                depth = xyz[:, :, 2]

                self.viz_pc(depth, rgb, xyz)  # 30-40ms

                now = perf_counter()
                dt = now - last_t
                print(f'dt: {dt:.4f}')
                last_t = now

    def viz_pc(self, depth, rgb, xyz):
        xyz_flat = xyz.reshape(-1, 3)
        is_valid = ~np.isnan(xyz_flat).any(axis=1)
        valid_idxs = np.where(is_valid)[0]
        xyz_flat_filtered = xyz_flat[valid_idxs]  # remove NaNs

        rgb_flat = rgb.reshape(-1, 3)
        rgb_flat = rgb_flat[valid_idxs]  # remove NaNs

        # publish inputs
        self.rgb_pub.publish(ros2_numpy.msgify(Image, rgb, encoding='rgb8'))
        self.depth_pub.publish(ros2_numpy.msgify(Image, depth, encoding='32FC1'))

        # create record array with x, y, and z fields
        pc = np.concatenate([xyz_flat_filtered, rgb_flat], axis=1).T

        pc_msg = pc_np_to_pc_msg(pc, names='x,y,z,r,g,b', frame_id='world')
        self.pc_pub.publish(pc_msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_yml', type=Path)
    args = parser.parse_args()

    rclpy.init()

    n = ZividNode(args.settings_yml)
    with zivid.Application() as app:
        with app.connect_camera() as camera:
            n.run(camera)


if __name__ == "__main__":
    main()
