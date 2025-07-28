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
import datetime

CAMERA_FRAME = 'zivid_optical_frame'
MASK_THRESHOLD = 0.25


class ZividNode(Node):
    def __init__(self,
        camera,
        settings_yml: Optional[Path] = None,
        use_rgb: bool = True,
        use_depth: bool = True,
        use_point_cloud: bool = True,
    ):
        super().__init__('zivid_node')
        if settings_yml is not None:
            self.settings = zivid.Settings.load(settings_yml)
        else:
            self.settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])
            suggest_settings_parameters = zivid.capture_assistant.SuggestSettingsParameters(
                max_capture_time=datetime.timedelta(milliseconds=5000),
                ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.none,
            )
            self.settings = zivid.capture_assistant.suggest_settings(
                camera, suggest_settings_parameters
            )

        self.pc_pub = self.create_publisher(PointCloud2, '/zivid/pc', 10) if use_point_cloud else None
        self.rgb_pub = self.create_publisher(Image, '/zivid/rgb', 10) if use_rgb else None
        self.depth_pub = self.create_publisher(Image, '/zivid/depth', 10) if use_depth else None

    def run(self, camera):
        while rclpy.ok():
            last_t = perf_counter()
            frame = camera.capture(self.settings)
        # last_t = perf_counter()
        # while rclpy.ok():
            with camera.capture(self.settings) as frame:
                a = perf_counter()
                point_cloud = frame.point_cloud()
                xyz_mm = point_cloud.copy_data("xyz")
                srgb = point_cloud.copy_data("srgb")

                xyz = xyz_mm / 1000.0
                rgb = srgb[:, :, :3]
                depth = xyz[:, :, 2]
                self.viz_pc(depth, rgb, xyz)  # 30-40ms
                # print(f"Captured in {perf_counter() - a} seconds")

    def viz_pc(self, depth, rgb, xyz):
        xyz_flat = xyz.reshape(-1, 3)
        is_valid = ~np.isnan(xyz_flat).any(axis=1)
        valid_idxs = np.where(is_valid)[0]
        xyz_flat_filtered = xyz_flat[valid_idxs]  # remove NaNs

        rgb_flat = rgb.reshape(-1, 3)
        rgb_flat = rgb_flat[valid_idxs]  # remove NaNs

        # publish inputs
        if self.rgb_pub:
            self.rgb_pub.publish(ros2_numpy.msgify(Image, rgb, encoding='rgb8'))
        if self.depth_pub:
            self.depth_pub.publish(ros2_numpy.msgify(Image, depth, encoding='32FC1'))

        # create record array with x, y, and z fields
        if self.pc_pub:
            pc = np.concatenate([xyz_flat_filtered, rgb_flat], axis=1).T
            pc_msg = pc_np_to_pc_msg(pc, names='x,y,z,r,g,b', frame_id=CAMERA_FRAME)
            self.pc_pub.publish(pc_msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings_yml', type=str, default=None)
    args = parser.parse_args()

    rclpy.init()
    # settings_yml = "/home/zixuanh/ros2_ws/configs/Zivid2_Settings_Zivid_Two_M70_ParcelsReflective.yml"
    # settings_yml = "/home/zixuanh/ros2_ws/configs/Zivid2_Settings_Zivid_Two_M70_ParcelsReflective_50Hz.yml"
    settings_yml = "/home/houhd/code/robot_tool_2025S/utils/ros_ws/config/zivid2_Settings_Zivid_Two_M70_ParcelsMatte_10Hz_4xsparse_enginetop_boxed.yml"

    app = zivid.Application()
    camera = app.connect_camera()
    n = ZividNode(
        camera,
        settings_yml,
        use_rgb=True,
        use_depth=False,
        use_point_cloud=False,
    )
    n.run(camera)


if __name__ == "__main__":
    main()
