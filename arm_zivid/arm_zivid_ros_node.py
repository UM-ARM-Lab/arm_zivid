import argparse

import cv2
from PIL import Image as PImage
import sys
sys.path.append('../')
from arm_robots_msgs.msg import GripperConstraint
from arm_robots_msgs.srv import SetGripperConstraints, SetGripperConstraintsRequest, SetGripperConstraintsResponse, SetCDCPDState, SetCDCPDStateRequest, SetCDCPDStateResponse
import numpy as np
import torch
import zivid
from arm_segmentation.predictor import Predictor
from cdcpd_torch.core.deformable_object_configuration import RopeConfiguration, DeformableObjectTracking
from cdcpd_torch.core.tracking_map import TrackingMap
from cdcpd_torch.data_utils.types.grippers import GrippersInfo, GripperInfoSingle
from cdcpd_torch.data_utils.types.point_cloud import PointCloud
from cdcpd_torch.modules.cdcpd_module_arguments import CDCPDModuleArguments
from cdcpd_torch.modules.cdcpd_network import CDCPDModule
from cdcpd_torch.modules.cdcpd_parameters import CDCPDParamValues
from cdcpd_torch.modules.post_processing.configuration import PostProcConfig, PostProcModuleChoice
from cdcpd_torch.core.visibility_prior import visibility_prior
from zivid.experimental import calibration
from rope_utils.cdcpd_helper import MujocoCDCPDDynamics
import ros_numpy
from ros_numpy.point_cloud2 import merge_rgb_fields
import rospy
import pickle
import pytorch_volumetric as pv
import pytorch_kinematics as pk

from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker

from arc_utilities.tf2wrapper import TF2Wrapper
from arc_utilities.listener import Listener
from scipy.interpolate import make_interp_spline, BSpline, splprep, splev
from time import perf_counter
import time
CAMERA_FRAME = 'zivid_optical_frame'
MASK_THRESHOLD = 0.25

class ExtensionCordConfiguration(RopeConfiguration):
    def __init__(self, num_points: int, max_rope_length: float, verts: torch.Tensor, dtype: torch.dtype = torch.double, device: torch.device = "cpu"):
        super().__init__(num_points, max_rope_length, rope_start_position=None, rope_end_position=None, dtype=dtype, device=device)
        self._verts_initial: torch.Tensor = verts

    def make_template(self) -> DeformableObjectTracking:
        # Make the edges
        template_edges = torch.zeros((2, self.num_points_ - 1),
                                     dtype=torch.long,
                                     device=self.device)
        template_edges[0, 0] = 0
        template_edges[1, -1] = self.num_points_ - 1
        for i in range(1, template_edges.shape[1]):
            template_edges[0, i] = i
            template_edges[1, i - 1] = i

        velocities = torch.zeros_like(self._verts_initial)

        configuration = DeformableObjectTracking()
        configuration.set_vertices(self._verts_initial)
        configuration.set_edges(template_edges)
        configuration.set_velocities(velocities)

        return configuration

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
    pc_msg = ros_numpy.msgify(PointCloud2, pc_rec, stamp=rospy.Time.now(), frame_id=frame_id)
    return pc_msg


def pairwise_squared_distances(a: torch.tensor, b: torch.tensor):
    a_s = np.sum(np.square(a), axis=-1, keepdims=True)  # [b, ..., n, 1]
    b_s = np.sum(np.square(b), axis=-1, keepdims=True)  # [b, ..., m, 1]
    dist = a_s - 2 * a @ np.moveaxis(b, -1, -2) + np.moveaxis(b_s, -1, -2)  # [b, ..., n, m]
    return dist

@torch.no_grad()
def cdcpd_helper(cdcpd_module: CDCPDModule, tracking_map: TrackingMap, depth: np.ndarray, mask: np.ndarray,
                 intrinsic: np.ndarray, grasped_points: GrippersInfo, seg_pc: torch.tensor, 
                 scene_sdf: pv.MeshSDF,kvis=100.0):
    """

    Args:
        cdcpd_module:
        tracking_map:
        depth:
        mask:
        intrinsic:
        grasped_points:
        seg_pc: [3, N]
        kvis:

    Returns:

    """
    verts = tracking_map.form_vertices_cloud()
    Y_emit_prior = None

    verts_np = verts.detach().numpy()
    depth[np.isnan(depth)] = 0
    Y_emit_prior = visibility_prior(verts_np, depth, mask, intrinsic[:3, :3], kvis/10)
    Y_emit_prior = torch.from_numpy(Y_emit_prior.reshape(-1, 1)).to(torch.double)
    assert torch.all(~torch.isnan(Y_emit_prior))

    # model_input.set_Y_emit_prior(Y_emit_prior)
    post_processing_out = cdcpd_module(
        verts,
        seg_pc,
        Y_emit_prior=Y_emit_prior,
        grippers_info=grasped_points,
        # scene_sdf=scene_sdf
    )
    tracking_map.update_def_obj_vertices(post_processing_out.detach())

    verts = tracking_map.form_vertices_cloud()
    cur_state_estimate = verts.detach().numpy().T

    return cur_state_estimate, Y_emit_prior


def estimate_to_msg(current_estimate):
    current_estimate_msg = MarkerArray()
    for i, p in enumerate(current_estimate):
        marker = Marker()
        marker.header.frame_id = CAMERA_FRAME
        marker.id = i
        marker.ns = f'point_{i}'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = p[0]
        marker.pose.position.y = p[1]
        marker.pose.position.z = p[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        current_estimate_msg.markers.append(marker)
    marker = Marker()
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.header.frame_id = CAMERA_FRAME
    marker.id = len(current_estimate)
    marker.scale.x = 0.01
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    for i, p in enumerate(current_estimate):
        marker.points.append(Point(*p))
    current_estimate_msg.markers.append(marker)
    return current_estimate_msg


class CDCPDTorchNode:
    """
    triggers the Capture service as fast as possible, and every time the data comes back (via topics),
    it runs the CDCPD algorithm and publishes the result.
    """

    def __init__(self):
        # CDCPD
        self.cdcpd_pred_pub = rospy.Publisher('cdcpd_pred', MarkerArray, queue_size=10)
        self.pc_pub = rospy.Publisher('pc', PointCloud2, queue_size=10)
        self.segmented_pc_pub = rospy.Publisher('segmented_pc', PointCloud2, queue_size=10)
        self.rgb_pub = rospy.Publisher('rgb', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('depth', Image, queue_size=10)
        self.mask_pub = rospy.Publisher('mask', Image, queue_size=10)
        self.spline_pub = rospy.Publisher('spline', MarkerArray, queue_size=10)
        self.gripper_constraints_srv = rospy.Service('set_gripper_constraints', SetGripperConstraints,
                                                     self.set_gripper_constraints)
        self.set_cdcpd_state_srv = rospy.Service('set_cdcpd_state', SetCDCPDState,
                                                 self.set_cdcpd_state)
        self.set_cdcpd_dyn_pred_srv = rospy.Service('set_cdcpd_dyn_pred', SetCDCPDState,
                                                 self.set_cdcpd_dyn_pred)
        self.tfw = TF2Wrapper()

        device = 'cpu'
        self.rope_start_pos_world_coords = torch.tensor([0.39, .8, .715]).double() #new
        rope_end_pos = torch.tensor(self.tfw.get_transform('zivid_optical_frame', 'right_tool'))[:3, -1].double()

        extrinsic = torch.tensor(self.tfw.get_transform('zivid_optical_frame', 'hdt_michigan_root')).double()
        extrinsic_inv = torch.tensor(self.tfw.get_transform('hdt_michigan_root', 'zivid_optical_frame')).double()
        self.extrinsic = extrinsic
        self.extrinsic_inv = extrinsic_inv
        #Convert self.rope_start_pos_world_coords to camera frame
        self.rope_start_pos = extrinsic @ torch.cat([self.rope_start_pos_world_coords, torch.tensor([1.0])]).double()
        self.rope_start_pos = self.rope_start_pos[:3] / self.rope_start_pos[3]
        # Segmentation
        self.seg_pred = Predictor(model_path='/home/peter/Documents/arm_segmentation/model.pth')

        self.NUM_TRACKED_POINTS = 25
        MAX_ROPE_LENGTH = 1.13 - .15
        # MAX_ROPE_LENGTH = 1.3
        USE_DEBUG_MODE = False
        # ALPHA = 5
        # BETA = 1.0
        # LAMBDA = 1.0
        # ZETA = 4.0

        # ALPHA = 250
        # BETA = 1
        # LAMBDA = 1.5
        # ZETA = 1

        ALPHA = 1.5
        BETA = .5
        LAMBDA = 1.0
        ZETA = 3

        W = .1
        OBSTACLE_COST_WEIGHT = 1
        # FIXED_POINTS_WEIGHT = 100.0
        # FIXED_POINTS_WEIGHT = 10.0
        FIXED_POINTS_WEIGHT = 5.0
        OBJECTIVE_VALUE_THRESHOLD = 1.0
        MIN_DISTANCE_THRESHOLD = -.02

        # sdfs = []
        # for i in range(20):
        rob_tf = pk.Transform3d(matrix=extrinsic_inv.cpu(), device='cpu')
        obj = pv.MeshObjectFactory("../data/assets/table_partial_decomp.obj")
        self.sdf = pv.MeshSDF(obj)

        self.sdf = pv.ComposedSDF([self.sdf], rob_tf)
        self.sdf.set_transforms(rob_tf, batch_dim=[1])

        # from cdcpd_torch.modules.debug_config import DebugConfig
        # cdcpd_debug_config = DebugConfig(is_verbose=True, is_profiling=True, is_using_additional_checks=False)

        self.gripper_constraints = [GripperConstraint('right_tool', self.NUM_TRACKED_POINTS - 1)]

        # TODO(Abhinav): Load in the vertices from Mujoco.
        self.verts_mujoco: torch.Tensor = torch.tensor(
            [[ 0.3526293  ,-0.58963396 , 1.60301896],
              [ 0.34301907, -0.54928824,  1.62530591],
              [ 0.33512668, -0.50815067,  1.64680576],
              [ 0.3293773 , -0.4661073 ,  1.66720526],
              [ 0.32621744, -0.42304167,  1.6859725 ],
              [ 0.32622005, -0.3789634 ,  1.70252387],
              [ 0.33013184, -0.33410223,  1.71627209],
              [ 0.33885917, -0.28899607,  1.72657293],
              [ 0.35341786, -0.24462921,  1.73261171],
              [ 0.37485684, -0.20271506,  1.73325575],
              [ 0.40399724, -0.16627531,  1.72694528],
              [ 0.4403917 , -0.1403643 ,  1.71208247],
              [ 0.48026281, -0.13092584,  1.68888657],
              [ 0.51721644, -0.13956355,  1.66101795],
              [ 0.54742801, -0.1619062 ,  1.63264706],
              [ 0.57071478, -0.19256557,  1.60554411],
              [ 0.5881991 , -0.22764377,  1.57945467],
              [ 0.60107567, -0.2644391 ,  1.55305131],
              [ 0.61037872, -0.30086926,  1.52471164],
              [ 0.61699974, -0.33529273,  1.49327874],
              [ 0.62170834, -0.36667522,  1.45849644],
              [ 0.62513038, -0.39472194,  1.42083331],
              [ 0.62771554, -0.41969782,  1.38100413],
              [ 0.62972947, -0.44209113,  1.33963598],
              [ 0.63124452, -0.46232661,  1.29714983]]
        ).T.double()
        #convert to world frame
        verts_mujoco_world = extrinsic_inv @ torch.cat([self.verts_mujoco, torch.ones(1, self.verts_mujoco.shape[1])], dim=0).double()
        z_offset = .4 * (torch.arange(25)/24)**.5
        x_offset = -.08 * (torch.arange(25)/24)**.5
        self.verts_init = verts_mujoco_world.clone()
        self.verts_init[2, :] -= z_offset
        self.verts_init[0, :] -= x_offset
        # Convert to camera frame
        self.verts_init = extrinsic @ self.verts_init.double()
        self.verts_init = self.verts_init[:3]
        self.verts_mujoco = self.verts_init.detach()
        def_obj_config = RopeConfiguration(self.NUM_TRACKED_POINTS, MAX_ROPE_LENGTH, self.rope_start_pos, rope_end_pos)
        # def_obj_config = ExtensionCordConfiguration(self.NUM_TRACKED_POINTS, MAX_ROPE_LENGTH, self.verts_mujoco)
        def_obj_config.initialize_tracking()

        self.previous_estimate = def_obj_config.initial_.vertices_.detach().numpy().T
        self.cdcpd_pred_pub.publish(estimate_to_msg(self.previous_estimate))

        self.tracking_map = TrackingMap()
        self.tracking_map.add_def_obj_configuration(def_obj_config)
        from cdcpd_torch.modules.post_processing.configuration import PostProcConstraintChoice, PostProcSolverChoice
        postproc_config = PostProcConfig(module_choice=PostProcModuleChoice.PRE_COMPILED, 
                                         gripper_constraint_choice=PostProcConstraintChoice.SOFT, 
                                         solver_choice=PostProcSolverChoice.GUROBI)

        param_vals = CDCPDParamValues()
        # GMM-EM Parameters
        param_vals.alpha_.val = ALPHA
        param_vals.beta_.val = BETA
        param_vals.lambda_.val = LAMBDA
        param_vals.zeta_.val = ZETA
        param_vals.w_.val = W
        param_vals.lle_neighbors.val = 8
        param_vals.gamma_.val = 2
        # Post-processing parameters
        param_vals.obstacle_cost_weight_.val = OBSTACLE_COST_WEIGHT
        param_vals.obstacle_constraint_min_dist_threshold.val = MIN_DISTANCE_THRESHOLD
        param_vals.fixed_points_weight_.val = FIXED_POINTS_WEIGHT
        

        print("Compiling CDCPD module...")
        # model = NN(False, .85, True)
        #Extract rotation and translation matrices from extrinsic_inv
        # Might need to fix if I am seeing weird dynamics behavior
        rotation_extrinsic_inv = extrinsic_inv[:3, :3]
        translation_extrinsic_inv = extrinsic_inv[:3, 3]
        self.dynamics_model = MujocoCDCPDDynamics(rotation_extrinsic_inv, translation_extrinsic_inv)
        cdcpd_module = CDCPDModule(deformable_object_config_init=def_obj_config, param_vals=param_vals,
                                   postprocessing_option=postproc_config, device=device,

                                #    dynamics_model=self.dynamics_model
                                   )#, debug_config=cdcpd_debug_config)
        self.cdcpd_module = cdcpd_module.eval()
        print("done.")

        self.intrinsics = None
        self.filter_radius = .1

    def get_grippers_info(self):
        """
        Assumes that node 0 is fixed to the wall at `self.rope_start_pos`, and any other constraints come from the
        `self.gripper_constraints` list.
        """
        grippers_info = GrippersInfo()
        grippers_info.append(GripperInfoSingle(fixed_pt_pred=self.rope_start_pos, grasped_vertex_idx=0))
        for c in self.gripper_constraints:
            pos = torch.tensor(self.tfw.get_transform('zivid_optical_frame', c.tf_frame_name))[:3, -1].double()
            grippers_info.append(GripperInfoSingle(fixed_pt_pred=pos, grasped_vertex_idx=c.cdcpd_node_index))

        return grippers_info

    def set_cdcpd_state(self, req: SetCDCPDStateRequest):
        # verts = self.tracking_map.form_vertices_cloud()
        # cur_state_estimate = verts.detach().numpy().T
        # print('cur_state_estimate', cur_state_estimate)
        # obj_vertices_np = np.array([ros_numpy.numpify(v) for v in req.vertices]).T
        # #Convert from world frame to camera frame
        # print('obj_vertices_np', obj_vertices_np)
        # print('extrinsic', self.extrinsic)
        # # obj_vertices_torch = torch.tensor(obj_vertices_np).double()
        # obj_vertices_torch = self.extrinsic @ torch.cat([torch.tensor(obj_vertices_np), torch.ones(1, obj_vertices_np.shape[1])], dim=0).double()
        # obj_vertices_torch = obj_vertices_torch[:3] / obj_vertices_torch[3]
        # print('obj_vertices_torch', obj_vertices_torch)
        # print(f"Resetting CDCPD to given state")
        # self.tracking_map.update_def_obj_vertices(obj_vertices_torch[:3].detach())
        self.tracking_map.update_def_obj_vertices(self.verts_init.detach())
        
        return SetCDCPDStateResponse()

    def set_cdcpd_dyn_pred(self, req: SetCDCPDStateRequest):
        obj_vertices_np = np.array([ros_numpy.numpify(v) for v in req.vertices]).T
        #Convert from world frame to camera frame
        # obj_vertices_torch = torch.tensor(obj_vertices_np).double()
        obj_vertices_torch = self.extrinsic @ torch.cat([torch.tensor(obj_vertices_np), torch.ones(1, obj_vertices_np.shape[1])], dim=0).double()
        obj_vertices_torch = obj_vertices_torch[:3] / obj_vertices_torch[3]
        print(f"Setting CDCPD dynamics to given state")
        self.dynamics_model.update_actual_state(obj_vertices_torch[:3].detach())
        return SetCDCPDStateResponse()

    def set_gripper_constraints(self, req: SetGripperConstraintsRequest):
        c: GripperConstraint
        self.gripper_constraints = req.constraints
        print(f"Got new gripper constraints: {self.gripper_constraints}")
        return SetGripperConstraintsResponse()

    def run(self, camera, cdcpd_iters):
        n_saved = 0
        cdcpd_iter_save = []
        this_run_ts = time.perf_counter()
        # self.tracking_map.update_def_obj_vertices(self.verts_init.detach())
        # self.previous_estimate = self.verts_init.detach().numpy().T
        # current_estimate_msg = estimate_to_msg(self.previous_estimate)
        # self.cdcpd_pred_pub.publish(current_estimate_msg)
        # print(self.verts_init)
        # print(self.verts_mujoco)
        # settings = zivid.Settings.load('camera_settings_initial_capture_ral.yml')
        settings = zivid.Settings.load('camera_settings_icra.yml')
        camera_matrix = calibration.intrinsics(camera, settings).camera_matrix
        self.intrinsics = np.array([
            [camera_matrix.fx, 0, camera_matrix.cx],
            [0, camera_matrix.fy, camera_matrix.cy],
            [0, 0, 1]
        ])
        # settings = zivid.Settings.load('camera_settings_icra.yml')


        last_t = perf_counter()
        itr = 0
        while True:
            with camera.capture(settings) as frame:
                a = time.perf_counter()
                capture_timestamp = a
                point_cloud = frame.point_cloud()
                xyz_mm = point_cloud.copy_data("xyz")
                rgba = point_cloud.copy_data("rgba")

                now = perf_counter()
                dt = now - last_t
                print(f'dt: {dt:.4f}')
                last_t = now

                xyz = xyz_mm / 1000.0
                rgb = rgba[:, :, :3]
                depth = xyz[:, :, 2]
                # print('depth', depth.shape)

                # save the rgb image every 100 frames, up to 10 times per run
                # if itr % 100 == 0:
                #     if n_saved < 10:
                #         PImage.fromarray(rgb).save(f"rgb_{int(now)}.png")
                #         n_saved += 1

                self.viz_pc(depth, rgb, xyz)  # 30-40ms
                # run segmentation. NOTE: this modifies the RGB buffer!
            
                mask, seg_pc = self.segmentation(rgb, xyz)  # 140ms
                if mask is None or seg_pc is None:
                    continue

                # Update grasped point
                grippers_info = self.get_grippers_info()

                # run cdcpd
                seg_pc_cdcpd = torch.from_numpy(seg_pc[:3, :]).double()  # CDCPD wants XYZ as a torch tensor

                self.segmented_pc_pub.publish(pc_np_to_pc_msg(seg_pc, names='x,y,z,r,g,b', frame_id=CAMERA_FRAME))

                b = time.perf_counter()
                print(f"Time for segmentation: {b - a:.4f}")
                a = time.time()
                if seg_pc.shape[1] > 0:
                    # try:
                    for _ in range(cdcpd_iters):
                        current_estimate, Y_emit_prior = cdcpd_helper(self.cdcpd_module, self.tracking_map, depth, mask,
                                                        self.intrinsics, grippers_info, seg_pc_cdcpd, self.sdf)  # 100ms
                        # Keep a running list of this and pickle it to send to Dylan
                        # cdcpd_iter_save.append((capture_timestamp, self.cdcpd_module._last_iter_results))
                    # except RuntimeError:
                    #     print("RuntimeError in cdcpd_helper!")
                    #     continue
                        
                    

                    b = time.time()
                    print('Time for cdcpd_helper:', b - a)
                    
                    current_estimate_msg = estimate_to_msg(current_estimate)
                    self.cdcpd_pred_pub.publish(current_estimate_msg)
                    # print(current_estimate)
                    self.previous_estimate = current_estimate
                    #Calculate a spline fit to the current estimate where Y_emit_prior > .5
                    # Y_emit_prior = Y_emit_prior.reshape(-1)
                    # current_estimate = np.array(current_estimate)
                    # current_estimate = current_estimate[Y_emit_prior > .5]
                    # print(Y_emit_prior)
                    # if current_estimate.shape[0] > 3:
                    #     x = current_estimate[:, 0]
                    #     y = current_estimate[:, 1]
                    #     z = current_estimate[:, 2]
                    #     tck, u = splprep([x, y, z], s=0.0)
                    #     u_new = np.linspace(0, 1, 100)
                    #     x_new, y_new, z_new = splev(u_new, tck)
                    #     spline_points = np.array([x_new, y_new, z_new])
                    #     spline_points = torch.from_numpy(spline_points).double()
                    #     spline_msg = estimate_to_msg(spline_points.T.detach().numpy())
                    #     self.spline_pub.publish(spline_msg)


            # if itr % 30 == 0:
            #     with open(f'./cdcpd_iter_save_{this_run_ts}.pkl', 'wb') as f:
            #         pickle.dump(cdcpd_iter_save, f)
            itr += 1

    def segmentation(self, rgb, xyz):
        # predictions = self.seg_pred.predict(rgb)  # 100ms
        
        # mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        # for p in predictions:
        #     if p['class'] == 'red_cable':
        #         binary_mask = (p['mask'] > MASK_THRESHOLD).astype(np.uint8)
        #         mask = cv2.bitwise_or(mask, binary_mask)
        # mask = mask.astype(bool)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv_mask = ((hsv[:, :, 0]) <= 10 | (hsv[:, :, 0] >= 220)) & (hsv[:, :, 2] >= 50)#& (hsv[:, : ,1] >= 150) & (hsv[:, :, 2] >= 150)
        # mask = mask & hsv_mask
        mask = hsv_mask
        rgb[~mask] = 0
        mask_msg = ros_numpy.msgify(Image, rgb, encoding='rgb8')
        self.mask_pub.publish(mask_msg)
        seg_xyz = xyz.copy()
        seg_xyz[~mask] = np.nan
        seg_pc = np.concatenate([seg_xyz, rgb], axis=-1)
        seg_pc_flat = seg_pc.reshape(-1, 6).T
        seg_pc_flat = seg_pc_flat[:, ~np.isnan(seg_pc_flat).any(axis=0)]

        too_few_points = seg_pc_flat.shape[1] < 100
        if too_few_points:
            print("Too few points in the segmented point cloud!")
            return None, None
        else:
            self.filter_radius = max(.1, self.filter_radius * 0.9)

        seg_pc_xyz = seg_pc_flat[:3, :]
        # remove all points that are far from the current estimate
        dists = np.sqrt(pairwise_squared_distances(seg_pc_xyz.T, self.previous_estimate))
        min_dists = dists.min(axis=1)
        near_idxs = np.where(min_dists < self.filter_radius)[0]
        seg_pc_near = seg_pc_flat[:, near_idxs]
        # print('Num points in filtered pc:', seg_pc_near.shape[1])

        #Convert seg_pc_near to world coordinates
        seg_pc_near_world = self.extrinsic_inv @ torch.cat([torch.tensor(seg_pc_near[:3]), torch.ones(1, seg_pc_near.shape[1])], dim=0).double()
        # y_mask = ((seg_pc_near_world[1] > .835) & (seg_pc_near_world[2] > .6))
        y_mask =  (seg_pc_near_world[1] > .804)

        #Convert seg_pc_near to camera frame
        seg_pc_near = seg_pc_near[:, ~y_mask]


        try:
            seg_pc_down = PointCloud(seg_pc_near[:3, :], seg_pc_near[3:, :])
            seg_pc_down.downsample(voxel_size=.03)
            seg_pc_down = np.concatenate((seg_pc_down.xyz, seg_pc_down.rgb)).astype(np.float64)
        except:
            seg_pc_down = seg_pc_near

        return mask, seg_pc_down.astype(np.float32)

    def viz_pc(self, depth, rgb, xyz):
        xyz_flat = xyz.reshape(-1, 3)
        is_valid = ~np.isnan(xyz_flat).any(axis=1)
        valid_idxs = np.where(is_valid)[0]
        xyz_flat_filtered = xyz_flat[valid_idxs]  # remove NaNs

        #Convert xyz_flat_filtered to world coordinates
        xyz_flat_filtered_world = self.extrinsic_inv @ torch.cat([torch.tensor(xyz_flat_filtered.T), torch.ones(1, xyz_flat_filtered.shape[0])], dim=0).double()
        mask = (xyz_flat_filtered_world[0] > .8)
        xyz_flat_filtered_world = xyz_flat_filtered_world[:, ~mask]
        xyz_flat_filtered = xyz_flat_filtered_world.numpy().astype(np.float32).T[:, :3]
        rgb_flat = rgb.reshape(-1, 3)
        rgb_flat = rgb_flat[valid_idxs]  # remove NaNs
        rgb_flat = rgb_flat[~mask]
        # publish inputs
        self.rgb_pub.publish(ros_numpy.msgify(Image, rgb, encoding='rgb8'))
        self.depth_pub.publish(ros_numpy.msgify(Image, depth, encoding='32FC1'))
        # create record array with x, y, and z fields
        pc = np.concatenate([xyz_flat_filtered, rgb_flat], axis=1).T

        pc_msg = pc_np_to_pc_msg(pc, names='x,y,z,r,g,b', frame_id='world')
        self.pc_pub.publish(pc_msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('iters', type=int, default=2)
    args = parser.parse_args()
    rospy.init_node("cdcpd_torch_node")
    n = CDCPDTorchNode()
    with zivid.Application() as app:
        with app.connect_camera() as camera:
            n.run(camera, args.iters)


if __name__ == "__main__":
    main()
