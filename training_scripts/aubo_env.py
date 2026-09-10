import math
import threading
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any


import numpy as np
import gymnasium as gym
from gymnasium import spaces


import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from control_msgs.action import GripperCommand


import tf2_ros
from tf2_ros import TransformException


try:
    from ros_gz_interfaces.srv import SetEntityPose
    from ros_gz_interfaces.msg import Entity
    _HAS_ROS_GZ_INTERFACES = True
except ImportError:
    _HAS_ROS_GZ_INTERFACES = False


ARM_JOINT_NAMES = [
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
]
GRIPPER_JOINT_NAME = "dh_ag95_left_outer_knuckle_joint"
GRIPPER_RANGE = (0.0, 0.93)
GRIPPER_CLOSE_FOR_GRASP = 0.55


# Giới hạn khớp an toàn (rad), chặn các góc quặt ngược tự va chạm
SAFE_JOINT_LIMITS = {
    "shoulder_joint": (-2.2, 2.2),
    "upperArm_joint": (-0.9, 0.6),
    "foreArm_joint": (0.35, 2.2),
    "wrist1_joint": (-1.5, 1.5),
    "wrist2_joint": (0.2, 2.8),
    "wrist3_joint": (-3.0, 3.0),
}


# Tư thế Home chuẩn cho thao tác mặt bàn: Cánh tay vươn ra trước, kẹp chúi thẳng xuống
# Tránh hoàn toàn tư thế "cây nến" (all-zeros)
HOME_JOINT_POSITIONS = np.array([0.0, -0.25, 1.40, 0.0, 1.57, 0.0], dtype=np.float64)


ARM_JOINT_VEL_LIMIT = {
    "shoulder_joint": 3.1416,
    "upperArm_joint": 3.1416,
    "foreArm_joint": 2.5656,
    "wrist1_joint": 3.1416,
    "wrist2_joint": 3.1416,
    "wrist3_joint": 3.1416,
}


_T_ACCEL_TO_MAX_VEL = 0.3
ARM_JOINT_ACCEL_LIMIT = {
    jname: vel / _T_ACCEL_TO_MAX_VEL for jname, vel in ARM_JOINT_VEL_LIMIT.items()
}


# Ngưỡng phát hiện va chạm bàn: chỉ báo động khi đầu kẹp đâm sâu xuống dưới mặt bàn
TABLE_TOP_Z = 0.80
TABLE_COLLISION_Z_MARGIN = 0.10  # Nâng lên 0.10m để kẹp hạ tới 0.70m ôm sát chân hộp mà không bị phạt oan


# Ngưỡng phát hiện kẹt / tự va chạm được nới lỏng để tránh báo động giả do trễ truyền thông
STALL_TRACKING_ERROR_THRESHOLD = 0.48   # rad (~27.5 độ)
STALL_CONSECUTIVE_STEPS = 15            # Cần lệch liên tục 15 nhịp mới tính là kẹt


SAFETY_VIOLATION_PENALTY = -2.0   # Khớp với giá trị đang thực dùng (trước đây bị "chết" - khai báo -15.0 nhưng code hardcode -2.0)
DROP_PENALTY = -10.0



REACH_PROGRESS_SCALE = 20.0
REACH_DIST_SCALE = 0.3     
PLACE_PROGRESS_SCALE = 20.0
PLACE_DIST_SCALE = 0.3


JERK_PENALTY_SCALE = 0.02         
ACTION_PENALTY_SCALE = 0.001       


ARM_TRAJECTORY_TOPIC = "/arm_controller/joint_trajectory"
JOINT_STATES_TOPIC = "/joint_states"
GRIPPER_ACTION_NAME = "/gripper_controller/gripper_cmd"


OBJECT_POSE_TOPIC = "/model/target_box/pose"
WORLD_NAME = "pick_and_place_world"
SET_ENTITY_POSE_SERVICE = f"/world/{WORLD_NAME}/set_pose"
OBJECT_MODEL_NAME = "target_box"


WORLD_FRAME = "world"
GRASP_LINK_FRAME = "dh_ag95_grasp_link"


BOX_HALF_SIZE = 0.03


OBJECT_SAMPLE_XY_RANGE = ((0.35, 0.60), (0.15, 0.40))
TARGET_SAMPLE_XY_RANGE = ((0.35, 0.60), (-0.40, -0.15))


SUCCESS_DIST_THRESHOLD = 0.04
LIFT_HEIGHT_THRESHOLD = 0.04

class _JointVelAccelLimiter:
    def __init__(
        self,
        joint_names,
        vel_limits: Dict[str, float],
        accel_limits: Dict[str, float],
        safe_limits: Dict[str, Tuple[float, float]],
    ):
        self._names = list(joint_names)
        self._vel_limits = np.array([vel_limits[n] for n in self._names], dtype=np.float64)
        self._accel_limits = np.array([accel_limits[n] for n in self._names], dtype=np.float64)
        self._lo = np.array([safe_limits[n][0] for n in self._names], dtype=np.float64)
        self._hi = np.array([safe_limits[n][1] for n in self._names], dtype=np.float64)


    def step(
        self,
        action_delta_norm: np.ndarray,
        q_cmd_prev: np.ndarray,
        qdot_cmd_prev: np.ndarray,
        dt: float,
        max_delta_frac: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        desired_qdot = (
            np.asarray(action_delta_norm, dtype=np.float64)
            * self._vel_limits
            * float(max_delta_frac)
        )


        max_dqdot = self._accel_limits * dt
        qdot_cmd = np.clip(
            desired_qdot, qdot_cmd_prev - max_dqdot, qdot_cmd_prev + max_dqdot
        )
        qdot_cmd = np.clip(qdot_cmd, -self._vel_limits, self._vel_limits)


        q_cmd = q_cmd_prev + qdot_cmd * dt
        q_cmd = np.clip(q_cmd, self._lo, self._hi)


        return q_cmd, qdot_cmd


    def freeze(self, q_safe: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(q_safe, dtype=np.float64), np.zeros(len(self._names), dtype=np.float64)


class AuboPickAndPlaceEnv(gym.Env):
    metadata = {"render_modes": []}


    def __init__(
        self,
        node_name: str = "aubo_pick_place_env",
        control_hz: float = 20.0,
        max_episode_steps: int = 300,
        max_delta_frac: float = 0.35,
        ros_args: Optional[list] = None,
    ):
        super().__init__()


        self.control_hz = control_hz
        self.control_period = 1.0 / control_hz
        self.max_episode_steps = max_episode_steps
        self.max_delta_frac = max_delta_frac


        self._n_arm = len(ARM_JOINT_NAMES)
        self._action_dim = self._n_arm + 1


        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32
        )
        obs_dim = 6 + 6 + 1 + 1 + 3 + 4 + 3 + 4 + 3 + 4 + self._action_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )


        if not rclpy.ok():
            rclpy.init(args=ros_args)
        self._node = _AuboRosBridge(node_name=node_name)


        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()


        self._step_count = 0
        self._prev_action = np.zeros(self._action_dim, dtype=np.float32)
        self._action_history = deque(maxlen=3)
        self._target_pos = np.array([0.5, -0.3, TABLE_TOP_Z + BOX_HALF_SIZE], dtype=np.float64)
        self._target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._has_lifted = False
        # Lưu khoảng cách ở bước trc, phục vụ Progress Shaping (chống Lazy Agent)
        self._prev_dist_ee_obj: Optional[float] = None
        self._prev_dist_obj_target: Optional[float] = None


        self._motion_limiter = _JointVelAccelLimiter(
            ARM_JOINT_NAMES, ARM_JOINT_VEL_LIMIT, ARM_JOINT_ACCEL_LIMIT, SAFE_JOINT_LIMITS
        )
        self._q_cmd = HOME_JOINT_POSITIONS.copy()
        self._qdot_cmd = np.zeros(self._n_arm, dtype=np.float64)
        self._q_cmd_safe_prev = HOME_JOINT_POSITIONS.copy()
        self._stall_counter = 0


        self._node.wait_for_first_data(timeout_sec=10.0)


    def set_max_delta_frac(self, frac: float) -> None:
        self.max_delta_frac = float(np.clip(frac, 0.10, 1.0))


    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)


        self._step_count = 0
        self._prev_action = np.zeros(self._action_dim, dtype=np.float32)
        self._action_history.clear()
        self._has_lifted = False


        # 1) Đưa tay về tư thế Home chuẩn (cúi xuống bàn, không vươn lên trời)
        self._node.publish_arm_trajectory(HOME_JOINT_POSITIONS.tolist(), duration_sec=1.5)


        self._q_cmd = HOME_JOINT_POSITIONS.copy()
        self._qdot_cmd = np.zeros(self._n_arm, dtype=np.float64)
        self._q_cmd_safe_prev = HOME_JOINT_POSITIONS.copy()
        self._stall_counter = 0


        # 2) Mở kẹp
        self._node.send_gripper_goal(position=GRIPPER_RANGE[0], max_effort=20.0)


        # 3) Đặt lại vị trí ngẫu nhiên cho vật thể
        #rng = self.np_random
        #obj_xy = (
        #    rng.uniform(*OBJECT_SAMPLE_XY_RANGE[0]),
         #   rng.uniform(*OBJECT_SAMPLE_XY_RANGE[1]),
        #)
        #tgt_xy = (
         #   rng.uniform(*TARGET_SAMPLE_XY_RANGE[0]),
         #   rng.uniform(*TARGET_SAMPLE_XY_RANGE[1]),
        #)
        #Phần bị chú thích vì đang thực hiện giai đoạn 1 của curriculum cho aubo gắp vật vs vị trí cố định
        fixed_obj_x = 0.90
        fixed_obj_y = -0.40
        fixed_tgt_x = 0.80
        fixed_tgt_y = 0.40


        obj_pose = Pose()
        obj_pose.position.x = float(fixed_obj_x)
        obj_pose.position.y = float(fixed_obj_y)
        obj_pose.position.z = float(TABLE_TOP_Z + BOX_HALF_SIZE)
        obj_pose.orientation.w = 1.0


        self._node.reset_object_pose(OBJECT_MODEL_NAME, obj_pose)


        self._target_pos = np.array(
            [fixed_tgt_x, fixed_tgt_y, TABLE_TOP_Z + BOX_HALF_SIZE], dtype=np.float32
        )
        self._target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


        # 4) Chờ chuyển động kết thúc hoàn toàn (1.6s > 1.5s duration)
        time.sleep(1.6)

        ee_pos0, _ = self._node.get_ee_pose(WORLD_FRAME, GRASP_LINK_FRAME)
        obj_pos0, _ = self._node.get_object_pose()
        self._prev_dist_ee_obj = float(
            np.linalg.norm(np.asarray(ee_pos0, dtype=np.float64) - np.asarray(obj_pos0, dtype=np.float64))
        )
        self._prev_dist_obj_target = float(
            np.linalg.norm(np.asarray(obj_pos0, dtype=np.float64) - self._target_pos.astype(np.float64))
        )


        obs = self._get_obs()
        info = {"is_success": False}
        return obs, info


    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


        gripper_now, _ = self._node.get_gripper_state()
        arm_delta_raw = action[: self._n_arm]
        gripper_delta_raw = float(action[self._n_arm])


        # Lấy trạng thái góc khớp thực tế làm gốc để chặn đứng tích lũy sai số trôi lệnh (Anti-windup)
        q_now, _ = self._node.get_arm_state()
        q_cmd, self._qdot_cmd = self._motion_limiter.step(
            action_delta_norm=arm_delta_raw,
            q_cmd_prev=np.asarray(q_now, dtype=np.float64),
            qdot_cmd_prev=self._qdot_cmd,
            dt=self.control_period,
            max_delta_frac=self.max_delta_frac,
        )
        self._q_cmd = q_cmd


        gripper_max_step = 2.1 * self.control_period * self.max_delta_frac
        gripper_cmd = float(
            np.clip(
                gripper_now + gripper_delta_raw * gripper_max_step,
                GRIPPER_RANGE[0],
                GRIPPER_RANGE[1],
            )
        )


        self._node.publish_arm_trajectory(
            q_cmd.tolist(), duration_sec=self.control_period * 1.5
        )
        if abs(gripper_cmd - gripper_now) > 1e-3:
            self._node.send_gripper_goal(position=gripper_cmd, max_effort=40.0)


        # Đồng bộ bước điều khiển
        time.sleep(self.control_period)


        # SAFETY WATCHDOG: Bỏ qua 8 bước đầu để hệ thống ổn định gia tốc
        safety_violation = None
        if self._step_count > 8:
            safety_violation = self._check_safety_violation(q_cmd)


        if safety_violation is not None:
            freeze_q, freeze_qdot = self._motion_limiter.freeze(self._q_cmd_safe_prev)
            self._node.publish_arm_trajectory(freeze_q.tolist(), duration_sec=0.5)
            self._q_cmd = freeze_q
            self._qdot_cmd = freeze_qdot
            self._stall_counter = 0
        else:
            self._q_cmd_safe_prev = q_cmd.copy()


        self._action_history.append(action.copy())
        obs = self._get_obs()
        reward, terminated, info = self._compute_reward(action, safety_violation)
        self._prev_action = action.copy()


        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps


        # Ngắt sớm nếu cánh tay vung quá cao khỏi bàn (chặn tật co tay lên trời)
        ee_pos = obs[14:17]
        if ee_pos[2] > 1.35:
            truncated = True


        return obs, reward, terminated, truncated, info


    def _check_safety_violation(self, q_cmd: np.ndarray) -> Optional[str]:
        def _get_frame_z(frame_candidates) -> Optional[float]:
            for frame_name in frame_candidates:
                try:
                    tf = self._node._tf_buffer.lookup_transform(
                        WORLD_FRAME, frame_name, rclpy.time.Time()
                    )
                    return float(tf.transform.translation.z)
                except Exception:
                    pass
            return None


        ee_pos, _ = self._node.get_ee_pose(WORLD_FRAME, GRASP_LINK_FRAME)
        obj_pos, _ = self._node.get_object_pose()


        # Lọc bỏ giá trị TF lỗi (0, 0, 0), tránh báo động giả
        # SỬA LỖI 1: Nới rộng vùng kiểm tra xuống -1.0 để không bỏ sót các pha kẹp đâm sâu xuống dưới bàn
        if ee_pos[2] > -1.0:
            if float(ee_pos[2]) < (TABLE_TOP_Z - TABLE_COLLISION_Z_MARGIN):
                return "table_collision"


        # Kiểm tra cao độ đốt cổ tay (wrist1) và khuỷu tay (foreArm)
        wrist_z = _get_frame_z(["wrist1_Link", "wrist1_link", "wrist2_Link", "wrist2_link"])
        if wrist_z is not None and wrist_z > 0.1:
            if wrist_z < (TABLE_TOP_Z + 0.02):
                return "table_collision"


        forearm_z = _get_frame_z(["foreArm_Link", "forearm_link"])
        if forearm_z is not None and forearm_z > 0.1:
            if forearm_z < (TABLE_TOP_Z + 0.05):
                return "table_collision"


        q_measured, _ = self._node.get_arm_state()
        tracking_error = np.abs(np.asarray(q_measured, dtype=np.float64) - q_cmd)
        if np.any(tracking_error > STALL_TRACKING_ERROR_THRESHOLD):
            self._stall_counter += 1
        else:
            self._stall_counter = 0


        if self._stall_counter >= STALL_CONSECUTIVE_STEPS:
            return "stall_or_collision"


        return None


    def close(self):
        try:
            self._executor.shutdown()
            self._node.destroy_node()
        finally:
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass


    def _get_obs(self) -> np.ndarray:
        q, qdot = self._node.get_arm_state()
        gripper_pos, gripper_vel = self._node.get_gripper_state()
        ee_pos, ee_quat = self._node.get_ee_pose(WORLD_FRAME, GRASP_LINK_FRAME)
        obj_pos, obj_quat = self._node.get_object_pose()


        obs = np.concatenate(
            [
                np.asarray(q, dtype=np.float32),
                np.asarray(qdot, dtype=np.float32),
                np.array([gripper_pos], dtype=np.float32),
                np.array([gripper_vel], dtype=np.float32),
                np.asarray(ee_pos, dtype=np.float32),
                np.asarray(ee_quat, dtype=np.float32),
                np.asarray(obj_pos, dtype=np.float32),
                np.asarray(obj_quat, dtype=np.float32),
                self._target_pos,
                self._target_quat,
                self._prev_action,
            ]
        )
        return obs.astype(np.float32)


    def _compute_reward(
        self, action: np.ndarray, safety_violation: Optional[str] = None
    ) -> Tuple[float, bool, Dict[str, Any]]:
        q, _ = self._node.get_arm_state()
        gripper_pos, _ = self._node.get_gripper_state()
        ee_pos, _ = self._node.get_ee_pose(WORLD_FRAME, GRASP_LINK_FRAME)
        obj_pos, _ = self._node.get_object_pose()


        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        obj_pos = np.asarray(obj_pos, dtype=np.float64)
        target_pos = self._target_pos.astype(np.float64)


        dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))
        dist_obj_target = float(np.linalg.norm(obj_pos - target_pos))


        is_closed = gripper_pos > GRIPPER_CLOSE_FOR_GRASP
        is_grasping = is_closed and dist_ee_obj < 0.06
        lifted = is_grasping and (obj_pos[2] - (TABLE_TOP_Z + BOX_HALF_SIZE)) > LIFT_HEIGHT_THRESHOLD
        was_lifted_before = self._has_lifted
        self._has_lifted = self._has_lifted or lifted


        #  (a) TIẾP CẬN: Progress Shaping 
       
        if self._prev_dist_ee_obj is None:
            self._prev_dist_ee_obj = dist_ee_obj  # bảo vệ nếu step() gọi trước reset() lần nào đó
           
        # SỬA LỖI 2: Chỉ thưởng tiến lại gần hộp nếu mũi kẹp vẫn ở trên mặt bàn an toàn, phạt nếu chìm xuống
        if ee_pos[2] >= (TABLE_TOP_Z - 0.02):
            r_reach_progress = REACH_PROGRESS_SCALE * (self._prev_dist_ee_obj - dist_ee_obj)
            r_reach = r_reach_progress - REACH_DIST_SCALE * dist_ee_obj
        else:
            r_reach = -2.0
           
        self._prev_dist_ee_obj = dist_ee_obj


        #  (b) Đặt vật về đích: Progress Shaping 
        if self._prev_dist_obj_target is None:
            self._prev_dist_obj_target = dist_obj_target
        if is_grasping or self._has_lifted:
            r_place_progress = PLACE_PROGRESS_SCALE * (self._prev_dist_obj_target - dist_obj_target)
            r_place = r_place_progress - PLACE_DIST_SCALE * dist_obj_target
        else:
            r_place = 0.0
        self._prev_dist_obj_target = dist_obj_target


        #(c) Thưởng khi nhấc bổng được vật
        r_lift_bonus = 10.0 if (lifted and not was_lifted_before) else (2.0 if lifted else 0.0)


        #(d) Thành công 
        success = self._has_lifted and dist_obj_target < SUCCESS_DIST_THRESHOLD and not is_closed
        r_success = 100.0 if success else 0.0


        #  (e) Phạt jerk 
        r_jerk = 0.0
        if len(self._action_history) == 3:
            a0, a1, a2 = self._action_history
            jerk = a2 - 2.0 * a1 + a0
            r_jerk = -JERK_PENALTY_SCALE * float(np.sum(np.square(jerk)))


        # (f) Phạt biên khớp an toàn (chỉ phạt khi thực sự vượt biên) 
        r_limit = 0.0
        for i, jname in enumerate(ARM_JOINT_NAMES):
            lo, hi = SAFE_JOINT_LIMITS[jname]
            over_hi = max(0.0, q[i] - hi)
            over_lo = max(0.0, lo - q[i])
            r_limit -= 10.0 * (over_hi + over_lo) ** 2


        #  (g) Phạt biên độ hành động 
        r_action_penalty = -ACTION_PENALTY_SCALE * float(np.sum(np.square(action)))


        # (h) Phạt rơi vật
        is_valid = not np.allclose(obj_pos, 0.0)
        out_of_z = obj_pos[2] < (TABLE_TOP_Z - 0.20)
        dropped = bool(is_valid and (self._step_count > 2) and out_of_z)
        r_drop_penalty = DROP_PENALTY if dropped else 0.0


        #(i) Phạt thời gian
        r_time = -0.05
       
      
        is_fatal_collision = (safety_violation == "table_collision")


        # (j) Phạt vi phạm an toàn 
        # Áp dụng phạt tột khung nếu đâm bàn để dập tắt hy vọng vơ vét điểm, nếu lỗi nhẹ thì giữ nguyên
        r_safety_penalty = -25.0 if is_fatal_collision else (SAFETY_VIOLATION_PENALTY if safety_violation is not None else 0.0)

        reward = (
            r_reach
            + r_place
            + r_lift_bonus
            + r_success
            + r_jerk
            + r_limit
            + r_action_penalty
            + r_drop_penalty
            + r_safety_penalty
            + r_time
        )
        terminated = bool(success or dropped or is_fatal_collision)


        info = {
            "is_success": success,
            "dist_ee_obj": dist_ee_obj,
            "dist_obj_target": dist_obj_target,
            "is_grasping": is_grasping,
            "dropped": dropped,
            "safety_violation": safety_violation,
            "max_delta_frac": self.max_delta_frac,
        }
        return float(reward), terminated, info




class _AuboRosBridge(Node):
    def __init__(self, node_name: str = "aubo_pick_place_env"):
        super().__init__(node_name)


        self._cb_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()


        self._q = HOME_JOINT_POSITIONS.copy()
        self._qdot = np.zeros(len(ARM_JOINT_NAMES), dtype=np.float64)
        self._gripper_pos = 0.0
        self._gripper_vel = 0.0
        self._object_pos = np.array([0.5, 0.3, TABLE_TOP_Z + BOX_HALF_SIZE], dtype=np.float64)
        self._object_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


        # Mặc định vị trí đầu kẹp ban đầu ở độ cao an toàn 1.0m (trên mặt bàn 0.8m)
        self._last_ee_pos = np.array([0.5, 0.0, 1.0], dtype=np.float64)
        self._last_ee_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


        self._joint_state_event = threading.Event()
        self._object_pose_event = threading.Event()


        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )


        self.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self._joint_state_cb,
            qos_sensor,
            callback_group=self._cb_group,
        )
        self.create_subscription(
            Pose,
            OBJECT_POSE_TOPIC,
            self._object_pose_cb,
            qos_sensor,
            callback_group=self._cb_group,
        )


        self._arm_traj_pub = self.create_publisher(
            JointTrajectory, ARM_TRAJECTORY_TOPIC, 10
        )


        self._gripper_client = ActionClient(
            self, GripperCommand, GRIPPER_ACTION_NAME, callback_group=self._cb_group
        )
        self._gripper_goal_handle = None


        self._set_pose_client = None
        if _HAS_ROS_GZ_INTERFACES:
            self._set_pose_client = self.create_client(
                SetEntityPose, SET_ENTITY_POSE_SERVICE, callback_group=self._cb_group
            )


        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)


    def _joint_state_cb(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        with self._lock:
            for i, jname in enumerate(ARM_JOINT_NAMES):
                if jname in name_to_idx:
                    idx = name_to_idx[jname]
                    if idx < len(msg.position):
                        self._q[i] = msg.position[idx]
                    if idx < len(msg.velocity):
                        self._qdot[i] = msg.velocity[idx]
            if GRIPPER_JOINT_NAME in name_to_idx:
                idx = name_to_idx[GRIPPER_JOINT_NAME]
                if idx < len(msg.position):
                    self._gripper_pos = msg.position[idx]
                if idx < len(msg.velocity):
                    self._gripper_vel = msg.velocity[idx]
        self._joint_state_event.set()


    def _object_pose_cb(self, msg: Pose):
        with self._lock:
            self._object_pos = np.array(
                [msg.position.x, msg.position.y, msg.position.z], dtype=np.float64
            )
            self._object_quat = np.array(
                [
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w,
                ],
                dtype=np.float64,
            )
        self._object_pose_event.set()


    def get_arm_state(self):
        with self._lock:
            return self._q.copy(), self._qdot.copy()


    def get_gripper_state(self):
        with self._lock:
            return float(self._gripper_pos), float(self._gripper_vel)


    def get_object_pose(self):
        with self._lock:
            return self._object_pos.copy(), self._object_quat.copy()


    def get_ee_pose(self, source_frame: str, target_frame: str):
        try:
            tf = self._tf_buffer.lookup_transform(
                source_frame, target_frame, rclpy.time.Time()
            )
            pos = np.array(
                [
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                ],
                dtype=np.float64,
            )
            quat = np.array(
                [
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z,
                    tf.transform.rotation.w,
                ],
                dtype=np.float64,
            )
            with self._lock:
                self._last_ee_pos = pos
                self._last_ee_quat = quat
            return pos, quat
        except TransformException:
            # Luôn trả về vị trí an toàn đã biết, KHÔNG BAO GIỜ trả về 0.0 để tránh dính bẫy va chạm
            with self._lock:
                pos = self._last_ee_pos.copy()
                quat = self._last_ee_quat.copy()
            return pos, quat


    def publish_arm_trajectory(self, q_target: list, duration_sec: float):
        msg = JointTrajectory()
        msg.joint_names = list(ARM_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q_target]
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        point.time_from_start.sec = sec
        point.time_from_start.nanosec = nanosec
        msg.points = [point]
        self._arm_traj_pub.publish(msg)


    def send_gripper_goal(self, position: float, max_effort: float = 40.0):
        if not self._gripper_client.wait_for_server(timeout_sec=0.5):
            return
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)


        if self._gripper_goal_handle is not None:
            try:
                self._gripper_goal_handle.cancel_goal_async()
            except Exception:
                pass


        future = self._gripper_client.send_goal_async(goal)
        future.add_done_callback(self._gripper_goal_response_cb)


    def _gripper_goal_response_cb(self, future):
        try:
            self._gripper_goal_handle = future.result()
        except Exception:
            pass


    def reset_object_pose(self, model_name: str, pose: Pose):
        if not _HAS_ROS_GZ_INTERFACES or self._set_pose_client is None:
            return
        if not self._set_pose_client.wait_for_service(timeout_sec=0.5):
            return


        req = SetEntityPose.Request()
        req.entity = Entity()
        req.entity.name = model_name
        req.entity.type = Entity.MODEL
        req.pose = pose


        # Chờ bất đồng bộ nhẹ nhàng, không dùng spin_until_future_complete gây xung đột luồng
        future = self._set_pose_client.call_async(req)
        start_t = time.time()
        while not future.done() and (time.time() - start_t) < 0.5:
            time.sleep(0.01)

    def wait_for_first_data(self, timeout_sec: float = 10.0):
        self._joint_state_event.wait(timeout_sec)
        self._object_pose_event.wait(timeout_sec)
