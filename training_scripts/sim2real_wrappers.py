"""

Thứ tự khi lồng wrapper (từ trong ra ngoài):

    env = AuboPickAndPlaceEnv(...)
    env = SensorNoiseWrapper(env)      # làm nhiễu quan sát TRƯỚC
    env = ActionLatencyWrapper(env)    # rồi mới trễ hành động
    env = Monitor(env)                 # (stable-baselines3)


"""

from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

# 1) SENSOR NOISE WRAPPER
# Chỉ số (index) các nhóm trong observation 42 chiều, PHẢI khớp với
# `AuboPickAndPlaceEnv._get_obs()` trong aubo_env.py. Nếu bạn đổi cấu
# trúc observation ở đó, hãy cập nhật lại các slice dưới đây
_IDX_Q = slice(0, 6)
_IDX_QDOT = slice(6, 12)
_IDX_GRIPPER_POS = 12
_IDX_GRIPPER_VEL = 13
_IDX_EE_POS = slice(14, 17)
_IDX_EE_QUAT = slice(17, 21)
_IDX_OBJECT_POS = slice(21, 24)
_IDX_OBJECT_QUAT = slice(24, 28)
# _IDX_TARGET_POS = slice(28, 31)   # KHÔNG thêm nhiễu 
# _IDX_TARGET_QUAT = slice(31, 35) # KHÔNG thêm nhiễu
# _IDX_PREV_ACTION = slice(35, 42) # KHÔNG thêm nhiễu

# Độ lệch chuẩn (std) mặc định cho từng nhóm, đơn vị tương ứng với obs
# (rad, rad/s, mét). Các giá trị này là ước lượng hợp lý cho encoder giá
# rẻ / ước lượng pose bằng thị giác - chỉnh lại cho khớp cảm biến thật
# của bạn khi có datasheet.
DEFAULT_NOISE_STD: Dict[str, float] = {
    "joint_pos": 0.01,      # rad      (~0.57 deg)  - nhiễu encoder khớp tay
    "joint_vel": 0.02,      # rad/s               - nhiễu vi phân vận tốc
    "gripper_pos": 0.005,   # rad                 - encoder kẹp
    "gripper_vel": 0.01,    # rad/s
    "ee_pos": 0.002,        # m  (2 mm)           - sai số FK/hiệu chuẩn
    "ee_quat": 0.005,       # (đơn vị quaternion, xem ghi chú bên dưới)
    "object_pos": 0.004,    # m  (4 mm)           - sai số ước lượng pose
                             #                       vật thể bằng camera
    "object_quat": 0.01,    # (đơn vị quaternion)
}


class SensorNoiseWrapper(gym.ObservationWrapper):
    """Thêm nhiễu Gaussian vào các thành phần "cảm biến" của observation.

    Tham số
    ----------
    env : gym.Env
        Môi trường gốc (hoặc wrapper khác), phải có observation dạng
        vector phẳng, layout giống `AuboPickAndPlaceEnv` (42 chiều).
    noise_std : dict, optional
        Ghi đè std cho 1 vài nhóm (các nhóm không nêu sẽ dùng mặc định
        trong DEFAULT_NOISE_STD). Đặt std = 0.0 để tắt nhiễu cho 1 nhóm
        cụ thể.
    enabled_groups : set[str], optional
        Nếu cung cấp, CHỈ những nhóm có tên trong set này mới bị làm
        nhiễu (các nhóm khác giữ nguyên bất kể std). Mặc định: tất cả
        các nhóm trong DEFAULT_NOISE_STD đều bật.
    seed : int, optional
        Seed riêng cho bộ sinh số ngẫu nhiên của nhiễu (độc lập với
        seed của môi trường / action space) để có thể tái lập kết quả.
    """

    def __init__(
        self,
        env: gym.Env,
        noise_std: Optional[Dict[str, float]] = None,
        enabled_groups: Optional[set] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env)

        obs_dim = int(np.prod(self.observation_space.shape))
        if obs_dim != 42:
            # Không raise cứng để không phá vỡ các biến thể env tương lai,
            # nhưng cảnh báo rõ vì các hằng số slice ở trên giả định 42
            gym.logger.warn(
                f"[SensorNoiseWrapper] observation_space có {obs_dim} chiều, "
                "khác 42 chiều mặc định của AuboPickAndPlaceEnv. Hãy kiểm "
                "tra lại các hằng số _IDX_* trong sim2real_wrappers.py."
            )

        self._std = dict(DEFAULT_NOISE_STD)
        if noise_std:
            self._std.update(noise_std)

        self._enabled = (
            set(enabled_groups) if enabled_groups is not None else set(self._std.keys())
        )

        self._rng = np.random.default_rng(seed)

   
    def _std_of(self, group: str) -> float:
        return self._std[group] if group in self._enabled else 0.0

    def _add_noise(self, values: np.ndarray, std: float) -> np.ndarray:
        if std <= 0.0:
            return values
        noise = self._rng.normal(loc=0.0, scale=std, size=values.shape)
        return values + noise.astype(values.dtype)

    def _add_quat_noise(self, quat: np.ndarray, std: float) -> np.ndarray:
        """Thêm nhiễu nhỏ vào quaternion rồi CHUẨN HOÁ lại (renormalize)
        để kết quả vẫn là 1 quaternion đơn vị hợp lệ. Đây là một xấp xỉ
        đơn giản cho nhiễu xoay nhỏ - đủ tốt để robust hoá chính sách RL,
        không nhằm mô phỏng chính xác phân phối SO(3)."""
        if std <= 0.0:
            return quat
        noisy = quat + self._rng.normal(loc=0.0, scale=std, size=quat.shape)
        norm = np.linalg.norm(noisy)
        if norm < 1e-8:
            return quat  # phòng trường hợp suy biến hiếm gặp
        return (noisy / norm).astype(quat.dtype)

    # ------------------------------------------------------------------
    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.array(observation, copy=True)

        obs[_IDX_Q] = self._add_noise(obs[_IDX_Q], self._std_of("joint_pos"))
        obs[_IDX_QDOT] = self._add_noise(obs[_IDX_QDOT], self._std_of("joint_vel"))

        obs[_IDX_GRIPPER_POS] = self._add_noise(
            obs[_IDX_GRIPPER_POS : _IDX_GRIPPER_POS + 1], self._std_of("gripper_pos")
        )[0]
        obs[_IDX_GRIPPER_VEL] = self._add_noise(
            obs[_IDX_GRIPPER_VEL : _IDX_GRIPPER_VEL + 1], self._std_of("gripper_vel")
        )[0]

        obs[_IDX_EE_POS] = self._add_noise(obs[_IDX_EE_POS], self._std_of("ee_pos"))
        obs[_IDX_EE_QUAT] = self._add_quat_noise(obs[_IDX_EE_QUAT], self._std_of("ee_quat"))

        obs[_IDX_OBJECT_POS] = self._add_noise(obs[_IDX_OBJECT_POS], self._std_of("object_pos"))
        obs[_IDX_OBJECT_QUAT] = self._add_quat_noise(
            obs[_IDX_OBJECT_QUAT], self._std_of("object_quat")
        )

        return obs.astype(np.float32)


# 2) ACTION LATENCY WRAPPER (hàng đợi FIFO trễ 20-60ms)

class ActionLatencyWrapper(gym.Wrapper):
    """Mô phỏng độ trễ ngẫu nhiên khi gửi Action xuống ROS 2 / phần cứng.

    Cơ chế:
        - Mỗi lần `step(action)` được gọi, action MỚI được đẩy vào 1
          hàng đợi FIFO (`collections.deque`) kèm theo mốc bước mà nó
          sẽ sẵn sàng để thực thi (được tính từ độ trễ ngẫu nhiên trong
          [min_ms, max_ms], quy đổi ra số bước dựa theo `control_hz`).
        - Trước khi gọi `env.step()` thật sự, wrapper lấy ra (pop) mọi
          action trong hàng đợi đã "đến hạn" (mốc bước <= bước hiện tại),
          và dùng action MỚI NHẤT trong số đó (các action cũ hơn bị lỗi
          thời/ghi đè - đúng với hành vi 1 buffer lệnh thực tế).
        - Nếu chưa có action nào đến hạn (giai đoạn "khởi động" của
          buffer), wrapper giữ nguyên (zero-order hold) action khả dụng
          gần nhất đã áp dụng.

    Tham số
    ----------
    env : gym.Env
        Môi trường gốc, cần có thuộc tính `control_hz` (số Hz điều khiển)
        để quy đổi mili-giây trễ sang số bước rời rạc. Nếu không tìm
        thấy, phải truyền `control_hz` tường minh.
    latency_range_ms : tuple(float, float)
        Khoảng trễ ngẫu nhiên (mili-giây), mặc định (20.0, 60.0).
    control_hz : float, optional
        Ghi đè tần số điều khiển nếu không lấy được từ `env.unwrapped`.
    seed : int, optional
        Seed riêng cho việc lấy mẫu độ trễ ngẫu nhiên.
    """

    def __init__(
        self,
        env: gym.Env,
        latency_range_ms: Tuple[float, float] = (20.0, 60.0),
        control_hz: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env)

        if latency_range_ms[0] < 0 or latency_range_ms[1] < latency_range_ms[0]:
            raise ValueError(
                f"latency_range_ms không hợp lệ: {latency_range_ms}"
            )
        self._latency_range_ms = latency_range_ms

        hz = control_hz if control_hz is not None else getattr(
            self.unwrapped, "control_hz", None
        )
        if hz is None or hz <= 0:
            raise ValueError(
                "Không xác định được `control_hz`. Hãy truyền tham số "
                "`control_hz=...` tường minh khi khởi tạo ActionLatencyWrapper."
            )
        self._control_period_ms = 1000.0 / hz

        # Hàng đợi FIFO: mỗi phần tử là (ready_step_idx, action_ndarray).
        self._queue: deque = deque()
        self._current_step = 0
        self._last_applied_action: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._queue.clear()
        self._current_step = 0
        self._last_applied_action = np.zeros(
            self.action_space.shape, dtype=self.action_space.dtype
        )
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=self.action_space.dtype)

        # 1) Lấy mẫu độ trễ ngẫu nhiên (ms) cho action lần này -> quy đổi
        #    sang số bước rời rạc (làm tròn, tối thiểu 1 bước để đảm bảo
        #    hành động không đi tắt tới hiện tại nếu latency rất nhỏ)
        latency_ms = self._rng.uniform(*self._latency_range_ms)
        delay_steps = max(1, int(round(latency_ms / self._control_period_ms)))
        ready_step_idx = self._current_step + delay_steps
        self._queue.append((ready_step_idx, action.copy()))

        # 2) Lấy ra action ms nhất đã thực thi tại bước hiện tại.
        #    Các action đến hạn cũ hơn bị bỏ (đã lỗi thời trong buffer thật)
        applied_action = self._last_applied_action
        while self._queue and self._queue[0][0] <= self._current_step:
            _, applied_action = self._queue.popleft()
        self._last_applied_action = applied_action

        self._current_step += 1

        # 3) Thực thi action đã trễ (không phải action vừa nhận) xuống
        #    môi trường gốc / ROS 2
        obs, reward, terminated, truncated, info = self.env.step(applied_action)
        info = dict(info)
        info["latency_ms_sampled"] = latency_ms
        return obs, reward, terminated, truncated, info


# Hàm tiện ích:
def apply_sim2real_wrappers(
    env: gym.Env,
    noise_std: Optional[Dict[str, float]] = None,
    latency_range_ms: Tuple[float, float] = (20.0, 60.0),
    seed: Optional[int] = None,
) -> gym.Env:
    """Bọc `env` lần lượt qua SensorNoiseWrapper rồi ActionLatencyWrapper.
    Dùng trong train_sac.py để giữ code khởi tạo môi trường ngắn gọn"""
    env = SensorNoiseWrapper(env, noise_std=noise_std, seed=seed)
    env = ActionLatencyWrapper(env, latency_range_ms=latency_range_ms, seed=seed)
    return env
