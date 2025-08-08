import argparse
from queue import Queue
from pathlib import Path
from typing import Optional
import threading

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
        num_workers: int = 2,
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

        self.frame_queue = Queue()
        self.processed_queue = Queue()
        self.num_workers = num_workers
        self.shutdown_event = threading.Event()
        self.workers = []
        self.publisher_thread = None

    def process_worker(self):
        """Worker thread that processes frames from the frame queue"""
        while not self.shutdown_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:  # Shutdown signal
                    break
                self.process_frame(frame)
                self.frame_queue.task_done()
            except:
                continue  # Timeout, continue loop

    def publisher_worker(self):
        """Publisher thread that publishes processed messages and times the frequency"""
        last_time = perf_counter()
        publish_count = 0

        while not self.shutdown_event.is_set():
            try:
                messages = self.processed_queue.get(timeout=1.0)
                if messages is None:  # Shutdown signal
                    break
                image_msg, depth_msg, pc_msg = messages
                self.pub2ros(image_msg, depth_msg, pc_msg)
                self.processed_queue.task_done()

                # Measure frequency
                publish_count += 1
                current_time = perf_counter()
                elapsed_time = current_time - last_time
                if elapsed_time >= 1.0:  # Log frequency every second
                    # self.get_logger().info(f"Publishing frequency: {publish_count / elapsed_time:.2f} Hz")
                    publish_count = 0
                    last_time = current_time
            except:
                continue  # Timeout, continue loop

    def run(self, camera):
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(target=self.process_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        # Start publisher thread
        self.publisher_thread = threading.Thread(target=self.publisher_worker, daemon=True)
        self.publisher_thread.start()

        # Main capture loop
        try:
            while rclpy.ok():
                frame = camera.capture(self.settings)
                self.frame_queue.put(frame)
        except KeyboardInterrupt:
            self.get_logger().info("Shutting down...")
        finally:
            self.shutdown()

    def process_frame(self, frame):
        a = perf_counter()
        point_cloud = frame.point_cloud()
        xyz_mm = point_cloud.copy_data("xyz")
        srgb = point_cloud.copy_data("srgb")

        xyz = xyz_mm / 1000.0
        rgb = srgb[:, :, :3]
        depth = xyz[:, :, 2]
        
        xyz_flat = xyz.reshape(-1, 3)
        is_valid = ~np.isnan(xyz_flat).any(axis=1)
        valid_idxs = np.where(is_valid)[0]
        xyz_flat_filtered = xyz_flat[valid_idxs]  # remove NaNs

        rgb_flat = rgb.reshape(-1, 3)
        rgb_flat = rgb_flat[valid_idxs]  # remove NaNs

        image_msg = ros2_numpy.msgify(Image, rgb, encoding='rgb8')
        depth_msg = ros2_numpy.msgify(Image, depth, encoding='32FC1')
        if self.pc_pub:
            pc = np.concatenate([xyz_flat_filtered, rgb_flat], axis=1).T
            pc_msg = pc_np_to_pc_msg(pc, names='x,y,z,r,g,b', frame_id=CAMERA_FRAME)

        self.processed_queue.put((image_msg, depth_msg, pc_msg if self.pc_pub else None))
        # self.get_logger().info(f"Captured in {perf_counter() - a} seconds")

    def pub2ros(self, image_msg, depth_msg, pc_msg):

        # publish inputs
        if self.rgb_pub:
            self.rgb_pub.publish(image_msg)
        if self.depth_pub:
            self.depth_pub.publish(depth_msg)
        if self.pc_pub:
            self.pc_pub.publish(pc_msg)

    def shutdown(self):
        """Gracefully shutdown all threads"""
        self.shutdown_event.set()
        
        # Send shutdown signals to queues
        for _ in range(self.num_workers):
            self.frame_queue.put(None)
        self.processed_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
        
        if self.publisher_thread:
            self.publisher_thread.join(timeout=2.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings_yml', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=2, help='Number of processing worker threads')
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
        num_workers=args.num_workers,
    )
    n.run(camera)


if __name__ == "__main__":
    main()
