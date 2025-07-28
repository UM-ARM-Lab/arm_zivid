import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create output directory
output_dir = "d405_capture_output"
os.makedirs(output_dir, exist_ok=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# D405 recommended resolution
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Align depth to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Create pointcloud object
pc = rs.pointcloud()
align = rs.align(rs.stream.color)

print("Starting camera... Press Ctrl+C to exit after capture.")
# info = rs.camera_info(0).serial_number
# print(f"Camera Serial Number: {info}")

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap to depth for visualization (optional)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # Show images side by side (optional)
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow("RGB and Depth", images)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Map the color to the depth
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        

        key = cv2.waitKey(1)
        if key == ord("c"):
            # Save color and depth images
            cv2.imwrite(os.path.join(output_dir, "color.png"), color_image)
            cv2.imwrite(os.path.join(output_dir, "depth_colormap.png"), depth_colormap)
            np.save(os.path.join(output_dir, "depth.npy"), depth_image)
            print("Saved RGB a  nd depth frames.")
            # Save as .ply (which you can later convert to .pcd using Open3D or PCL)
            ply_filename = "output_pointcloud.ply"
            points.export_to_ply(ply_filename, color_frame)
            print(f"Saved point cloud to {ply_filename}")
        elif key == 27:  # ESC key
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
