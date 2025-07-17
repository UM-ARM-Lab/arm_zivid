import re
from glob import glob
import os
from time import sleep
from tqdm import tqdm

import argparse
from pathlib import Path
from typing import Optional
from time import perf_counter
import datetime
from queue import Queue

# Parallel processing 
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zivid

import rclpy
from rclpy.node import Node
import psutil

from std_msgs.msg import Header
from arm_zivid.utils import get_local_hostname, store_data_dict, get_file_extension

CAMERA_FRAME = 'zivid_optical_frame'
MASK_THRESHOLD = 0.25

def get_local_hostname():
    import socket
    hostname = socket.getfqdn()
    if len(hostname) == 0:
        hostname = "Unknown"
    if not ".local" in hostname:
        hostname += ".local"
    return hostname

class ZividLocalNode(Node):

    def __init__(self,
            camera,
            dataset_root,
            dataset_name,
            config_name,  # New: config identifier
            settings_yml: Optional[Path] = None,
            chunk_size=50,      # Save every n images
            dry_run=False,
            process=True,  # If raw is not set, process online
            verbose=False,            # New: Enable verbose output
            output_format="h5",       # New: Output format (h5 or zarr)
        ):
        # Create unique node name with config identifier
        node_name = f'zivid_node_local_{config_name}' if config_name else 'zivid_node_local'
        super().__init__(node_name)
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

        self.camera = camera
        self.config_name = config_name
        self.online_processing = process
        self.verbose = verbose
        self.output_format = output_format
        
        # Make metadata publisher with config-specific topic
        topic_name = f"/zivid_node_local/frame_id" if config_name else "/zivid_node_local/frame_id"
        self.idx_pub = self.create_publisher(Header, topic_name, 10)

        # Make dataset collection with config-specific path
        self._make_ds_tmpl(dataset_root, dataset_name, config_name)
        
        self.frame_idx = 0
        self.dry_run = dry_run

        # Initialize post processor for online mode
        if self.online_processing:
            self.post_processor = ZividPostProcessor(
                dataset_root,
                dataset_name,
                config_name,  # Pass config_name
                chunk_size=chunk_size,
                dry_run=dry_run,
                online_mode=True,
                verbose=verbose,
                output_format=output_format
            )
        else:
            self.frame_queue = Queue()
            # Setup procesors for raw frame saving only
            self.save_executor = ThreadPoolExecutor(max_workers=8)
            self.saving_thread = threading.Thread(target=self.start_raw_save_daemon)

        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.shutdown_event = threading.Event()

    def _make_ds_tmpl(self, dataset_root, dataset_name, config_name):
        self.host = get_local_hostname()

        # Make dataset chunk names with config-specific subdirectory
        if config_name:
            path = os.path.abspath(os.path.join(dataset_root, dataset_name, config_name))
        else:
            path = os.path.abspath(os.path.join(dataset_root, dataset_name))
        os.makedirs(path, exist_ok=True)
        self.dataset_path = path
        
        # Use appropriate file extension based on output format
        file_ext = get_file_extension(self.output_format)
        dataset_tmpl = os.path.join(path, "processed_chunk")
        self.dataset_tmpl = dataset_tmpl + "_{0}" + file_ext
        
        # Raw frame template for online mode
        frame_tmpl = os.path.join(path, "frame")
        self.frame_tmpl = frame_tmpl + "_{0}.zdf"

    def _stamped_header(self, content, timestamp=None):
        if timestamp is None:
            timestamp = self.get_clock().now().to_msg()
        msg = Header()
        msg.stamp = timestamp
        msg.frame_id = content
        return msg

    def capture_loop(self):
        last_fps_ctr = perf_counter()
        nframes = 0
        print(f"🚀 Starting capture loop - Online processing: {self.online_processing}")
        while rclpy.ok() and not self.shutdown_event.is_set():
            frame = self.camera.capture(self.settings)
            timestamp = self.get_clock().now().to_msg()

            if self.online_processing:
                self.post_processor.add_frame_online(self.frame_idx, frame, timestamp)
            else:
                self.frame_queue.put((self.frame_idx, frame))

            # Publish frame_id (sequence id = frame count)
            if not rclpy.ok():
                break
            self.idx_pub.publish(self._stamped_header(
                self.host+":"+self.dataset_path+f"/{self.frame_idx}", timestamp
            ))
            self.frame_idx += 1

            # Debugging speed
            nframes += 1
            if perf_counter()-1.0 >= last_fps_ctr:
                if self.verbose:
                    # Get memory usage
                    memory = psutil.virtual_memory()
                    memory_used_gb = memory.used / (1024**3)
                    memory_total_gb = memory.total / (1024**3)
                    memory_percent = memory.percent
                    
                    if self.online_processing:
                        history_len = len(self.post_processor.frame_history)
                        processed_results = len(self.post_processor.processed_results)
                        rgb_queue_size = self.post_processor.rgb_queue.qsize()
                        print(f"📊 FPS: {nframes} | History: {history_len} | Pending: {processed_results} | RGB Queue: {rgb_queue_size} | Memory: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB ({memory_percent:.1f}%)")
                    else:
                        print(f"📊 FPS: {nframes} | Raw Queue: {self.frame_queue.qsize()} | Memory: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB ({memory_percent:.1f}%)")
                last_fps_ctr = perf_counter()
                nframes = 0

    def start_raw_save_daemon(self):
        """
        Loop process to save all incoming frames
        """
        while not self.shutdown_event.is_set():
            if not self.frame_queue.empty():
                (frame_id, frame) = self.frame_queue.get()
                self.save_executor.submit(self.raw_save_worker, frame_id, frame)

    def raw_save_worker(self, frame_id, frame):
        """Individual worker to save raw frame files"""
        filename = self.frame_tmpl.format(frame_id)
        if not self.dry_run:
            frame.save(filename)
        del frame

    def run(self):
        self.capture_thread.start()
        if self.online_processing:
            # Start post processor for online mode
            self.post_processor.run()
        else:
            # Start raw frame saving thread for offline mode
            self.saving_thread.start()
        rclpy.get_default_context().on_shutdown(self.shutdown)
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self)
        executor.spin()

    def shutdown(self):
        """Implement proper shutdown for proper frame saving"""
        self.get_logger().info(f"Waiting for frame processors to finish")
        self.shutdown_event.set()
        
        # Stop ROS spinning only if not already shutdown
        if rclpy.ok():
            rclpy.shutdown()
        
        self.capture_thread.join()
        
        if self.online_processing:
            self.post_processor.shutdown()
        else:
            self.saving_thread.join()
            self.save_executor.shutdown(wait=True)

    def get_latest_frames(self, n=1):
        """Get the latest n processed frames (only available in online mode)"""
        if self.online_processing:
            return self.post_processor.get_latest_frames(n)
        else:
            raise RuntimeError("Frame retrieval only available in online processing mode")

    def get_frame_by_id(self, frame_id):
        """Get a specific frame by ID (only available in online mode)"""
        if self.online_processing:
            return self.post_processor.get_frame_by_id(frame_id)
        else:
            raise RuntimeError("Frame retrieval only available in online processing mode")


class ZividPostProcessor:
    def __init__(self,
            dataset_root,
            dataset_name,
            config_name,  # New: config identifier
            chunk_size=50,      # Save every n images
            dry_run=False,
            online_mode=False,  # New: Enable online processing mode
            verbose=False,      # New: Enable verbose output
            output_format="h5", # New: Output format (h5 or zarr)
        ):
        # Make dataset collection
        self.config_name = config_name
        self.output_format = output_format
        self._make_ds_tmpl(dataset_root, dataset_name, config_name)
        
        self.chunk_idx = 0
        self.chunk_size = chunk_size
        self.dry_run = dry_run
        self.online_mode = online_mode
        self.verbose = verbose

        # Input queues
        self.frame_queue = Queue()  # For offline: (frame_id, file_path), online: (frame_id, frame_obj, timestamp)
        
        # Processing result queues (for ordered insertion)
        self.processed_results = {}  # frame_id -> (rgb, depth, pc, timestamp)
        self.next_expected_frame = 0  # For maintaining order
        self.results_lock = threading.Lock()
        
        # Final ordered queues for chunked storage
        self.rgb_queue = Queue()
        self.depth_queue = Queue()
        self.pc_queue = Queue()
        self.timestamps_queue = Queue()  # New: Store capture timestamps
        
        # In-memory history for online access
        self.frame_history = []  # List of (frame_id, rgb, depth, pc, timestamp)
        self.history_lock = threading.Lock()

        # Setup procesors
        self.processing_thread = threading.Thread(target=self.start_processing)
        self.ordering_thread = threading.Thread(target=self.start_ordering)  # New: Handles result ordering
        self.chunking_thread = threading.Thread(target=self.start_chunking)  # New: Handles chunk saving
        self.save_executor = ThreadPoolExecutor(max_workers=8)
        self.shutdown_event = threading.Event()

    def _make_ds_tmpl(self, dataset_root, dataset_name, config_name):
        self.host = get_local_hostname()

        # Make dataset chunk names with config-specific subdirectory
        if config_name:
            path = os.path.abspath(os.path.join(dataset_root, dataset_name, config_name))
        else:
            path = os.path.abspath(os.path.join(dataset_root, dataset_name))
        os.makedirs(path, exist_ok=True)
        self.dataset_path = path
        
        # Use appropriate file extension based on output format
        file_ext = get_file_extension(self.output_format)
        dataset_tmpl = os.path.join(path, "processed_chunk")
        self.dataset_tmpl = dataset_tmpl + "_{0}" + file_ext
        
        # Raw frame template for online mode
        frame_tmpl = os.path.join(path, "frame")
        self.frame_tmpl = frame_tmpl + "_{0}.zdf"

    def get_frames(self):
        """
        Get list of files that match the pattern ``self.dataset_path/frame_{idx}.zdf``, where idx must be a positive integer.
        Check if they are continuous and start with 0, and raise Exception if not.
        Otherwise, return list of files in order of frame_idx.
        """
        pattern = os.path.join(self.dataset_path, "frame_*.zdf")
        files = glob(pattern)

        # Extract indices and map to file
        frame_map = {}
        for f in files:
            match = re.search(r"frame_(\d+)\.zdf$", f)
            if match:
                idx = int(match.group(1))
                frame_map[idx] = f
        if not frame_map:
            raise Exception("No matching frames found.")

        sorted_indices = sorted(frame_map.keys())

        # Check continuity
        expected = list(range(sorted_indices[0], sorted_indices[-1] + 1))
        if sorted_indices != expected or sorted_indices[0] != 0:
            raise Exception(f"Non-continuous or non-zero-starting frame indices: {sorted_indices}")

        return [frame_map[i] for i in sorted_indices]

    def start_processing(self):
        while not self.shutdown_event.is_set():
            if not self.frame_queue.empty():
                frame_data = self.frame_queue.get()
                print(f"After get, {self.frame_queue.qsize()} frames in queue")
                if self.online_mode:
                    # Online mode: (frame_id, frame_obj, timestamp)
                    frame_id, frame_obj, timestamp = frame_data
                    self.save_executor.submit(self.post_process_worker, frame_id, frame_obj, timestamp)
                else:
                    # Offline mode: (frame_id, file_path) - convert to online format
                    frame_id, frame_file = frame_data
                    self.save_executor.submit(self.post_process_worker, frame_id, frame_file, None)
            else:
                sleep(0.01)

    def post_process_worker(self, frame_id, frame_source, timestamp=None):
        """Unified worker that processes frames from either objects (online) or files (offline)"""
        is_online = self.online_mode and timestamp is not None

        # Get frame object
        if isinstance(frame_source, str):
            # Offline mode: load from file
            frame_obj = zivid.Frame(frame_source)
            should_cleanup = True
        else:
            # Online mode: use provided frame object
            frame_obj = frame_source
            should_cleanup = True
        
        try:
            # Process frame (same for both modes)
            point_cloud = frame_obj.point_cloud()
            xyz_mm = point_cloud.copy_data("xyz")
            img = point_cloud.copy_data("srgb")

            xyz = xyz_mm / 1000.0
            rgb = img[:, :, :3]
            depth = xyz[:, :, 2]

            xyz_flat = xyz.reshape(-1, 3)
            is_valid = ~np.isnan(xyz_flat).any(axis=1)
            valid_idxs = np.where(is_valid)[0]
            xyz_flat_filtered = xyz_flat[valid_idxs]  # remove NaNs

            rgb_flat = rgb.reshape(-1, 3)
            rgb_flat = rgb_flat[valid_idxs]  # remove NaNs
            pc = np.concatenate([xyz_flat_filtered, rgb_flat], axis=1).T
        except:
            print("Error extracting data from frame")

        # Handle output based on mode
        if is_online:
            # Online mode: use ordered processing with timestamps
            with self.results_lock:
                self.processed_results[frame_id] = (rgb, depth, pc, timestamp)
        else:
            # Offline mode: use ordered processing but with None timestamp for consistency
            with self.results_lock:
                self.processed_results[frame_id] = (rgb, depth, pc, None)

        # Clean up frame object
        if should_cleanup:
            del frame_obj

    def start_ordering(self):
        """Thread to maintain ordered processing results"""
        while not self.shutdown_event.is_set():
            with self.results_lock:
                if self.next_expected_frame in self.processed_results:
                    rgb, depth, pc, timestamp = self.processed_results.pop(self.next_expected_frame)
                    
                    # Add to ordered queues
                    self.rgb_queue.put((self.next_expected_frame, rgb))
                    self.depth_queue.put((self.next_expected_frame, depth))
                    self.pc_queue.put((self.next_expected_frame, pc))
                    self.timestamps_queue.put((self.next_expected_frame, timestamp))
                    
                    # Add to in-memory history
                    with self.history_lock:
                        self.frame_history.append((self.next_expected_frame, rgb, depth, pc, timestamp))
                        # Keep history size manageable (last 100 frames)
                        if len(self.frame_history) > 100:
                            self.frame_history.pop(0)
                    
                    self.next_expected_frame += 1
            sleep(0.001)  # Short sleep to avoid busy waiting

    def start_chunking(self):
        """Thread to save chunks when they reach the specified size"""
        while not self.shutdown_event.is_set():
            if self.rgb_queue.qsize() >= self.chunk_size:
                self.save_chunk()
            sleep(0.1)  # Check every 100ms

    def add_frame_online(self, frame_id, frame_obj, timestamp):
        """Add frame for online processing"""
        self.frame_queue.put((frame_id, frame_obj, timestamp))

    def add_frame_offline(self, frame_id, frame_file):
        """Add frame for offline processing"""
        self.frame_queue.put((frame_id, frame_file))

    def get_latest_frames(self, n=1):
        """Get the latest n processed frames from history"""
        with self.history_lock:
            return self.frame_history[-n:] if len(self.frame_history) >= n else self.frame_history[:]

    def get_frame_by_id(self, frame_id):
        """Get a specific frame by ID from history"""
        with self.history_lock:
            for fid, rgb, depth, pc, timestamp in self.frame_history:
                if fid == frame_id:
                    return (fid, rgb, depth, pc, timestamp)
        return None

    def save_chunk(self):
        st = perf_counter()
        save_size = min(self.chunk_size, self.rgb_queue.qsize())

        # Concatenate
        rgb_arr = [self.rgb_queue.get() for _ in range(save_size)]
        depth_arr = [self.depth_queue.get() for _ in range(save_size)]
        pc_arr = [self.pc_queue.get() for _ in range(save_size)]

        # Get timestamps (both online and offline modes now use timestamps_queue)
        timestamp_arr = []
        if not self.timestamps_queue.empty():
            timestamp_arr = [self.timestamps_queue.get() for _ in range(min(save_size, self.timestamps_queue.qsize()))]

        # Sort all by frame_id
        rgb_arr.sort(key=lambda x: x[0])
        depth_arr.sort(key=lambda x: x[0])
        pc_arr.sort(key=lambda x: x[0])
        if timestamp_arr:
            timestamp_arr.sort(key=lambda x: x[0])

        # Stack data
        rgb_stacked = np.stack([e[1] for e in rgb_arr])
        depth_stacked = np.stack([e[1] for e in depth_arr])
        
        # Find max number of points across all point clouds
        max_num_pts = max(e[1].shape[1] for e in pc_arr)
        num_pcs = len(pc_arr)
        
        # Create pre-filled array with np.inf
        pc_stacked = np.full((num_pcs, 6, max_num_pts), np.inf, dtype=np.float32)
        
        # Copy each point cloud into the large array
        for i, (frame_id, pc) in enumerate(pc_arr):
            num_pts = pc.shape[1]
            pc_stacked[i, :, :num_pts] = pc

        # Make dict and save as specified format
        save_dict = {
            "rgb": rgb_stacked,
            "depth": depth_stacked,
            "pc": pc_stacked,
        }

        for k, v in save_dict.items():
            print(f"{k} shape: {v.shape}")
        
        # Add timestamps if available and not None (convert ROS Time to float for compatibility)
        if timestamp_arr:
            valid_timestamps = []
            for e in timestamp_arr:
                if e[1] is not None:  # Only add non-None timestamps
                    valid_timestamps.append(float(e[1].sec) + float(e[1].nanosec) * 1e-9)
                else:
                    valid_timestamps.append(0.0)  # Use 0.0 for offline frames
            if any(t > 0 for t in valid_timestamps):  # Only save if we have real timestamps
                save_dict["timestamps"] = valid_timestamps

        print("Took", perf_counter() - st, "seconds to add timestamp")

        # Save file using the specified output format
        save_path = self.dataset_tmpl.format(self.chunk_idx)
        if not self.dry_run:
            store_data_dict(save_path, save_dict, self.output_format)
        print(f"Saved {len(rgb_arr)} images to {save_path} (format: {self.output_format}) in {perf_counter() - st:.2f}s")
        self.chunk_idx += 1
        

    def run(self):
        """
        Start processing threads using unified worker for both online and offline modes
        """
        self.processing_thread.start()
        
        if self.online_mode:
            # Start ordering and chunking threads for online mode
            self.ordering_thread.start()
            self.chunking_thread.start()
            print("Online processing mode started")
        else:
            # Offline mode: use unified worker with ordering for consistency
            self.ordering_thread.start()
            self.chunking_thread.start()
            
            frame_files = self.get_frames()
            print(f"Starting offline processing of {len(frame_files)} frames with unified worker")
            
            with tqdm(total=len(frame_files), desc="Post-processing frames") as pbar:
                for frame_id, frame_file in enumerate(frame_files):
                    self.add_frame_offline(frame_id, frame_file)
                    pbar.update(1)
                
                # Wait for all frames to be processed and ordered
                print("Waiting for frames to be processed...")
                while self.next_expected_frame < len(frame_files):
                    sleep(0.01)
                    
                # Wait for chunks to be saved  
                print("Waiting for chunks to be saved...")
                while not self.rgb_queue.empty():
                    sleep(0.1)
                
            print(f"Processed {len(frame_files)} frames using unified worker")

    def shutdown(self):
        print("Shutting Down")
        self.shutdown_event.set()
        
        # Join all threads
        if self.processing_thread.is_alive():
            self.processing_thread.join()
        if hasattr(self, 'ordering_thread') and self.ordering_thread.is_alive():
            self.ordering_thread.join()
        if hasattr(self, 'chunking_thread') and self.chunking_thread.is_alive():
            self.chunking_thread.join()
            
        self.save_executor.shutdown(wait=True)

        # Final save if anything is left
        if not self.rgb_queue.empty():
            self.save_chunk()

def extract_config_name(settings_path):
    """Extract config name from settings file path (filename without .yml extension)"""
    if settings_path is None:
        return "default"
    return Path(settings_path).stem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["collect", "process"], default="collect")
    parser.add_argument("-s", "--settings_yml", type=Path, nargs='+', required=True,
        help="One or more settings YAML files")
    parser.add_argument("-r", "--dataset_root", type=Path, default=os.path.expanduser("~/datasets/zivid"))
    parser.add_argument("-n", "--dataset_name", type=str, default="test")
    parser.add_argument("-c", "--chunk_size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--raw", action="store_true", help="Save raw frames and process later")
    parser.add_argument("--timeout", type=float, default=None, help="Timeout in seconds for capture/processing (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output (frame dimensions, point cloud info, FPS stats)")
    parser.add_argument("--output-format", type=str, choices=["h5", "zarr"], default="h5",
        help="Output format for processed data chunks (h5 or zarr)")

    # Parse
    args = parser.parse_args()
    
    def timeout_handler(timeout_seconds, instances):
        """Handle timeout by triggering shutdown on all instances"""
        sleep(timeout_seconds)
        print(f"\n⏰ Timeout reached ({timeout_seconds}s), shutting down all instances...")
        for instance in instances:
            instance.shutdown()
    
    if args.task == "collect":
        rclpy.init()
        app = zivid.Application()
        camera = app.connect_camera()
        
        # Create instances for each settings file
        nodes = []
        for settings_path in args.settings_yml:
            config_name = extract_config_name(settings_path)
            print(f"🔧 Creating capture instance for config: {config_name} (format: {args.output_format})")
            
            node = ZividLocalNode(camera,
                        args.dataset_root,
                        args.dataset_name,
                        config_name,
                        settings_yml=settings_path,
                        chunk_size=int(args.chunk_size),
                        dry_run=args.dry_run,
                        process=not args.raw,  # If raw is not set, process online
                        verbose=args.verbose,
                        output_format=args.output_format
                    )
            nodes.append(node)
        
        # Start timeout thread if specified
        timeout_thread = None
        if args.timeout is not None:
            print(f"⏱️  Starting with {args.timeout}s timeout for {len(nodes)} instances")
            timeout_thread = threading.Thread(target=timeout_handler, args=(args.timeout, nodes))
            timeout_thread.daemon = True
            timeout_thread.start()
        
        # Start all nodes
        node_threads = []
        for i, node in enumerate(nodes):
            print(f"🚀 Starting capture instance {i+1}/{len(nodes)}: {node.config_name}")
            node_thread = threading.Thread(target=node.run)
            node_thread.daemon = True
            node_thread.start()
            node_threads.append(node_thread)
        
        try:
            # Spin ROS - all nodes share the same ROS context
            # rclpy.spin_once(nodes[0])  # Use first node for spinning
            while rclpy.ok():
                # for node in nodes:
                #     rclpy.spin_once(node, timeout_sec=0.01)
                sleep(0.01)
        except KeyboardInterrupt:
            print("\n🛑 KeyboardInterrupt received")
        except Exception as e:
            print(f"❌ Error during processing: {e}")
        finally:
            # Shutdown all nodes
            for i, node in enumerate(nodes):
                print(f"🛑 Shutting down instance {i+1}/{len(nodes)}: {node.config_name}")
                node.shutdown()
                node.destroy_node()
            
            # Wait for all threads to complete
            for thread in node_threads:
                thread.join(timeout=5.0)
            
            if timeout_thread and timeout_thread.is_alive():
                print("⏰ Timeout thread stopped")
            
    # Offline processing feature
    elif args.task == "process":
        processors = []
        
        # Create processors for each settings file
        for settings_path in args.settings_yml:
            config_name = extract_config_name(settings_path)
            print(f"🔧 Creating processor instance for config: {config_name} (format: {args.output_format})")
            
            processor = ZividPostProcessor(
                args.dataset_root,
                args.dataset_name,
                config_name,
                chunk_size=int(args.chunk_size),
                dry_run=args.dry_run,
                verbose=args.verbose,
                output_format=args.output_format
            )
            processors.append(processor)
        
        # Start timeout thread if specified
        timeout_thread = None
        if args.timeout is not None:
            print(f"⏱️  Starting with {args.timeout}s timeout for {len(processors)} instances")
            timeout_thread = threading.Thread(target=timeout_handler, args=(args.timeout, processors))
            timeout_thread.daemon = True
            timeout_thread.start()
        
        # Start all processors
        processor_threads = []
        for i, processor in enumerate(processors):
            print(f"🚀 Starting processor instance {i+1}/{len(processors)}: {processor.config_name}")
            processor_thread = threading.Thread(target=processor.run)
            processor_thread.start()
            processor_threads.append(processor_thread)
    
        try:
            # Wait for all processors to complete
            for i, thread in enumerate(processor_threads):
                print(f"⏳ Waiting for processor {i+1}/{len(processors)} to complete...")
                thread.join()
        except KeyboardInterrupt:
            print("\n🛑 KeyboardInterrupt received")
        except Exception as e:
            print(f"❌ Error during processing: {e}")
        finally:
            # Shutdown all processors
            for i, processor in enumerate(processors):
                print(f"🛑 Shutting down processor {i+1}/{len(processors)}: {processor.config_name}")
                processor.shutdown()

    else:
        print("Unrecognized task:", args.task)

if __name__ == "__main__":
    main()
