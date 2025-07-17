"""Hand-eye calibration sample."""
import datetime
import pdb
import pickle
import time

from arm_utilities.tf2wrapper import TF2Wrapper
from arm_utilities.transformation_helper import build_mat_from_transform
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import numpy as np
import zivid
import rclpy
import ros2_numpy
from sensor_msgs.msg import Image, PointCloud2
from arm_zivid.pc_np_to_pc_msg import pc_np_to_pc_msg
from threading import Thread


def _acquire_checkerboard_frame(camera):
    print("Capturing checkerboard image... ")

    suggest_settings_parameters = zivid.capture_assistant.SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=1200),
        ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.none,
    )

    settings = zivid.capture_assistant.suggest_settings(
        camera, suggest_settings_parameters
    )
    return camera.capture(settings)


def _enter_robot_pose(index):
    inputted = input(
        "Enter pose with id={} (a line with 16 space separated values describing 4x4 row-major matrix):".format(
            index
        )
    )
    elements = inputted.split(maxsplit=15)
    data = np.array(elements, dtype=np.float64).reshape((4, 4))
    robot_pose = zivid.calibration.Pose(data)
    print("The following pose was entered:\n{}".format(robot_pose))
    return robot_pose


CAMERA_FRAME = 'zivid_optical_frame'
ROBOT_FRAME = 'victor_left_tool0'


def _main():
    rclpy.init()
    node = Node("handeye_calibration_zivid")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin)
    spin_thread.start()
    app = zivid.Application()
    camera = app.connect_camera()
    tfwrapper = TF2Wrapper(node)
    current_pose_id = 0
    # calibration_inputs = pickle.load(open("latest_raw_inputs.pkl", "rb"))
    calibration_inputs = []
    board_in_cam_list = []
    robot_to_hand_transforms = []
    calibrate = False
    pc_pub = node.create_publisher(PointCloud2, 'pc', 10)
    rgb_pub = node.create_publisher(Image, 'rgb', 10)
    while not calibrate:
        command = input(
            "Enter command, p (to add robot pose) or c (to perform calibration):"
        ).strip()
        if command == "p":
            try:
                # robot_pose = _enter_robot_pose(current_pose_id)
                transform = tfwrapper.get_transform("victor_root", ROBOT_FRAME)
                robot_pose = build_mat_from_transform(transform)
                robot_pose[:3, 3] *= 1000
                robot_to_hand_transforms.append(robot_pose)
                robot_pose = zivid.calibration.Pose(robot_pose)
                print("The following pose was entered:\n{}".format(robot_pose))
                frame = _acquire_checkerboard_frame(camera)
                point_cloud = frame.point_cloud()
                xyz_mm = point_cloud.copy_data("xyz")
                rgb = point_cloud.copy_data("rgba")[..., :3]
                t1 = time.time()
                rgb_msg = ros2_numpy.msgify(Image, rgb, encoding='rgb8')
                print(f"Converting to ROS image took {time.time()-t1}")
                rgb_pub.publish(rgb_msg)
                print(f"Publishing RGB image took {time.time() - t1}")
                xyz = xyz_mm / 1000.0
                xyz_flat = xyz.reshape(-1, 3)
                is_valid = ~np.isnan(xyz_flat).any(axis=1)
                valid_idxs = np.where(is_valid)[0]
                xyz_flat_filtered = xyz_flat[valid_idxs]  # remove NaNs

                rgb_flat = rgb.reshape(-1, 3)
                rgb_flat = rgb_flat[valid_idxs]  # remove NaNs
                pc = np.concatenate([xyz_flat_filtered, rgb_flat], axis=1).T

                pc_msg = pc_np_to_pc_msg(pc, names='x,y,z,r,g,b', frame_id=CAMERA_FRAME)
                t2 = time.time()
                pc_pub.publish(pc_msg)
                print(f"Publishing point cloud took {time.time() - t2}")
                print("Detecting checkerboard square centers... ")
                result = zivid.calibration.detect_feature_points(point_cloud)

                if result:
                    print("OK")
                    board_in_cam = result.pose().to_matrix()
                    board_in_cam_list.append(board_in_cam)

                    res = zivid.calibration.HandEyeInput(robot_pose, result)
                    calibration_inputs.append(res)
                    # with open("latest_raw_inputs.pkl", "wb") as f:
                    #     pickle.dump(calibration_inputs, f)
                    current_pose_id += 1
                    if current_pose_id >= 2:
                        calibration_result = zivid.calibration.calibrate_eye_to_hand(calibration_inputs)
                        # calibration_result = zivid.calibration.calibrate_eye_in_hand(calibration_inputs)
                        calibrated_mat = calibration_result.transform()
                        calibrated_mat[:3, 3] /= 1000
                        print(calibrated_mat)
                        tfwrapper.send_transform_matrix(calibrated_mat,  "victor_root", CAMERA_FRAME, is_static=True)
                        if calibration_result:
                            print("OK")
                            print("Result:\n{}".format(calibration_result))
                        else:
                            print("FAILED")
                else:
                    print("FAILED")
            except ValueError as ex:
                print(ex)
        elif command == "c":
            calibrate = True
        else:
            print("Unknown command '{}'".format(command))

    print("Performing hand-eye calibration...")
    zivid.calibration.calibrate_eye_to_hand(calibration_inputs)
    calibration_result = zivid.calibration.calibrate_eye_in_hand(calibration_inputs)
    if calibration_result:
        print("OK")
        print("Result:\n{}".format(calibration_result))
    else:
        print("FAILED")


if __name__ == "__main__":
    _main()
