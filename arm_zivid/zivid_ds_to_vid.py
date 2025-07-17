import argparse
import os
import glob
import re
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

def natural_sort_key(filename):
    """Extract chunk number for natural sorting of processed_chunk_N files"""
    # Extract the base filename without path
    basename = os.path.basename(filename)
    
    # Look for pattern like "processed_chunk_123" or "chunk_123"
    match = re.search(r'chunk_(\d+)', basename)
    if match:
        return int(match.group(1))
    else:
        # Fallback to lexicographical sorting if no chunk number found
        return basename

def detect_dataset_format(input_dir):
    """Detect if the dataset uses h5 or zarr format by checking the first file"""
    h5_files = glob.glob(os.path.join(input_dir, "*.h5"))
    zarr_dirs = glob.glob(os.path.join(input_dir, "*.zarr"))
    
    if h5_files and zarr_dirs:
        raise ValueError(f"Found both h5 and zarr files in {input_dir}. Please use a directory with only one format.")
    elif h5_files:
        return "h5", sorted(h5_files, key=natural_sort_key)
    elif zarr_dirs:
        return "zarr", sorted(zarr_dirs, key=natural_sort_key)
    else:
        raise ValueError(f"No h5 or zarr files found in {input_dir}")

def load_h5_data(file_path, data_type):
    """Load data from h5 file"""
    import h5py
    
    with h5py.File(file_path, 'r') as f:
        if data_type in f:
            return np.array(f[data_type])
        else:
            raise KeyError(f"'{data_type}' not found in {file_path}")

def load_zarr_data(file_path, data_type):
    """Load data from zarr directory (zarr v2 compatible)"""
    import zarr
    
    # Ensure zarr v2 compatibility
    store = zarr.DirectoryStore(file_path)
    root = zarr.open_group(store=store, mode='r')
    
    if data_type in root:
        return np.array(root[data_type])
    else:
        raise KeyError(f"'{data_type}' not found in {file_path}")

def normalize_depth(depth_data, min_depth=None, max_depth=None):
    """Normalize depth data to 0-255 range for visualization"""
    if min_depth is None:
        # Filter out invalid values (NaN, inf, negative)
        valid_depth = depth_data[np.isfinite(depth_data) & (depth_data > 0)]
        if len(valid_depth) > 0:
            min_depth = np.percentile(valid_depth, 1)  # Use 1st percentile to avoid outliers
        else:
            min_depth = 0
    
    if max_depth is None:
        valid_depth = depth_data[np.isfinite(depth_data) & (depth_data > 0)]
        if len(valid_depth) > 0:
            max_depth = np.percentile(valid_depth, 99)  # Use 99th percentile to avoid outliers
        else:
            max_depth = 1
    
    # Clip and normalize
    depth_clipped = np.clip(depth_data, min_depth, max_depth)
    depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    
    # Handle invalid values by setting them to 0 (black)
    depth_normalized[~np.isfinite(depth_data) | (depth_data <= 0)] = 0
    
    return depth_normalized

def create_video_writer(output_path, fps, frame_size):
    """Create OpenCV video writer"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

def process_rgb_frames(frames_data, output_path, fps):
    """Process RGB frames and create video"""
    print(f"Processing {len(frames_data)} RGB frames...")
    
    # Get frame dimensions from first frame
    first_frame = frames_data[0]
    height, width = first_frame.shape[:2]
    
    # Create video writer
    writer = create_video_writer(output_path, fps, (width, height))
    
    try:
        for frame in tqdm(frames_data, desc="Writing RGB frames"):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()
    
    print(f"RGB video saved to: {output_path}")

def process_depth_frames(frames_data, output_path, fps, min_depth=None, max_depth=None):
    """Process depth frames and create video"""
    print(f"Processing {len(frames_data)} depth frames...")
    
    # Get frame dimensions from first frame
    first_frame = frames_data[0]
    height, width = first_frame.shape[:2]
    
    # Create video writer
    writer = create_video_writer(output_path, fps, (width, height))
    
    try:
        for frame in tqdm(frames_data, desc="Writing depth frames"):
            # Normalize depth to 0-255 range
            depth_normalized = normalize_depth(frame, min_depth, max_depth)
            
            # Convert to 3-channel grayscale for video
            depth_3ch = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
            writer.write(depth_3ch)
    finally:
        writer.release()
    
    print(f"Depth video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Zivid dataset to video")
    parser.add_argument("input_dir", type=Path, help="Input directory containing h5 or zarr files")
    parser.add_argument("-o", "--output", type=Path, help="Output video file path (default: auto-generated)")
    parser.add_argument("--data-type", choices=["rgb", "depth", "pc"], default="rgb",
                       help="Type of data to convert (default: rgb)")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Video frame rate (default: 30.0)")
    parser.add_argument("--min-depth", type=float, default=None,
                       help="Minimum depth value for normalization (depth only)")
    parser.add_argument("--max-depth", type=float, default=None,
                       help="Maximum depth value for normalization (depth only)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    # Check for point cloud (not implemented yet)
    if args.data_type == "pc":
        raise NotImplementedError("Point cloud video generation is not implemented yet")
    
    # Detect dataset format
    try:
        format_type, file_list = detect_dataset_format(args.input_dir)
        print(f"Detected dataset format: {format_type}")
        print(f"Found {len(file_list)} files")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Generate output path if not provided
    if args.output is None:
        output_name = f"{args.input_dir.name}_{args.data_type}_{format_type}.mp4"
        args.output = args.input_dir.parent / output_name
    
    # Load and concatenate all frames
    all_frames = []
    
    for file_path in tqdm(file_list, desc=f"Loading {format_type} files"):
        try:
            if format_type == "h5":
                frames = load_h5_data(file_path, args.data_type)
            else:  # zarr
                frames = load_zarr_data(file_path, args.data_type)
            
            # Add frames to the list
            for frame in frames:
                all_frames.append(frame)
                
        except KeyError as e:
            print(f"Warning: {e} in {file_path}, skipping...")
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not all_frames:
        print("No frames found to process!")
        return
    
    print(f"Total frames collected: {len(all_frames)}")
    
    # Process frames based on data type
    if args.data_type == "rgb":
        process_rgb_frames(all_frames, args.output, args.fps)
    elif args.data_type == "depth":
        process_depth_frames(all_frames, args.output, args.fps, args.min_depth, args.max_depth)
    
    print(f"Video generation complete!")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
