import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

try:
    from ros_gz_interfaces.srv import SpawnEntity, DeleteEntity
    from ros_gz_interfaces.msg import Entity
    _HAS_ROS_GZ_INTERFACES = True
except ImportError:  
    _HAS_ROS_GZ_INTERFACES = False



@dataclass
class RandomizedPhysics:
    """Tham số vật lý đã random hoá thành công cho 1 episode - hữu ích
    để log lại vào TensorBoard/CSV trong `train_sac.py`."""
    mass: float
    mu1: float
    mu2: float

_BOX_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{model_name}">
    <link name="link">
      <inertial>
        <mass>{mass:.6f}</mass>
        <inertia>
          <ixx>{ixx:.8f}</ixx>
          <iyy>{iyy:.8f}</iyy>
          <izz>{izz:.8f}</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box>
        </geometry>
        <surface>
          <friction>
            <ode><mu>{mu1:.4f}</mu><mu2>{mu2:.4f}</mu2></ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
    </link>
    <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
      <publish_model_pose>true</publish_model_pose>
      <publish_link_pose>false</publish_link_pose>
      <publish_nested_models>false</publish_nested_models>
      <use_pose_vector_msg>false</use_pose_vector_msg>
      <update_frequency>30</update_frequency>
    </plugin>
  </model>
</sdf>
"""


class DomainRandomizer:

    def __init__(
        self,
        world_name: str = "pick_and_place_world",
        model_name: str = "target_box",
        box_size: Tuple[float, float, float] = (0.06, 0.06, 0.06),
        mass_range: Tuple[float, float] = (0.05, 2.0),
        friction_range: Tuple[float, float] = (0.4, 1.5),
        service_timeout_sec: float = 5.0,
        node_name: str = "domain_randomizer_node",
        logger: Optional[logging.Logger] = None,
    ):
        if not _HAS_ROS_GZ_INTERFACES:
            raise RuntimeError(
                "Gói 'ros_gz_interfaces' chưa được cài đặt "
                "(vd: `sudo apt install ros-jazzy-ros-gz-interfaces`). "
                "Đây là gói bắt buộc để DomainRandomizer gọi được "
                "SpawnEntity/DeleteEntity."
            )
        if not rclpy.ok():
            raise RuntimeError(
                "rclpy chưa được init. Hãy gọi `rclpy.init()` trc khi "
                "tạo DomainRandomizer (train_sac.py đã làm việc này)."
            )

        self._world_name = world_name
        self._model_name = model_name
        self._box_size = box_size
        self._mass_range = mass_range
        self._friction_range = friction_range
        self._timeout = service_timeout_sec

        self._node = rclpy.create_node(node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._logger = logger or self._node.get_logger()

        create_srv = f"/world/{world_name}/create"
        remove_srv = f"/world/{world_name}/remove"
        self._spawn_client = self._node.create_client(SpawnEntity, create_srv)
        self._delete_client = self._node.create_client(DeleteEntity, remove_srv)

    # ------------------------------------------------------------------
    def randomize(self, pose: Pose) -> Optional[RandomizedPhysics]:
        """Xoá `target_box` hiện tại, sinh lại tại `pose` với mass/ma sát
        random. Trả về `RandomizedPhysics` nếu spawn thành công; trả về
        `None` nếu service timeout/lỗi (được log cảnh báo/lỗi, KHÔNG
        raise exception để không làm sập vòng lặp huấn luyện - episode
        vẫn tiếp tục, chỉ là không có domain randomization ở lần đó)."""
        mass = random.uniform(*self._mass_range)
        mu1 = random.uniform(*self._friction_range)
        mu2 = random.uniform(*self._friction_range)

        deleted = self._delete_entity()
        if not deleted:
            self._logger.warn(
                f"[DomainRandomizer] Xoá '{self._model_name}' thất bại hoặc "
                "timeout (có thể do model chưa tồn tại ở lần reset đầu "
                "tiên) - vẫn tiếp tục thử spawn."
            )

        sdf = self._build_sdf(mass, mu1, mu2)
        spawned = self._spawn_entity(sdf, pose)
        if not spawned:
            self._logger.error(
                f"[DomainRandomizer] Spawn '{self._model_name}' THẤT BẠI - "
                "vật thể có thể không tồn tại trong world ở episode này."
            )
            return None

        return RandomizedPhysics(mass=mass, mu1=mu1, mu2=mu2)

    # ------------------------------------------------------------------
    def _build_sdf(self, mass: float, mu1: float, mu2: float) -> str:
        sx, sy, sz = self._box_size
        # Mô-men quán tính xấp xỉ của khối hộp chữ nhật đặc, đồng chất -
        # đủ chính xác cho mục đích domain randomization.
        ixx = (mass / 12.0) * (sy**2 + sz**2)
        iyy = (mass / 12.0) * (sx**2 + sz**2)
        izz = (mass / 12.0) * (sx**2 + sy**2)
        return _BOX_SDF_TEMPLATE.format(
            model_name=self._model_name,
            mass=mass, ixx=ixx, iyy=iyy, izz=izz,
            mu1=mu1, mu2=mu2, sx=sx, sy=sy, sz=sz,
        )

    def _call_service_sync(self, client, request, timeout_sec: float):
        """Gọi service và chờ đồng bộ bằng `rclpy.spin_until_future_complete`
        trên node + executor RIÊNG của class này - ĐÚNG cơ chế `ros2
        service call` (CLI) dùng, đã được kiểm chứng hoạt động ổn định.
        An toàn 100% vì node này không bị executor nào khác spin cùng
        lúc (xem giải thích ở đầu file)."""
        t_start = time.monotonic()

        if not client.wait_for_service(timeout_sec=timeout_sec):
            self._logger.warn(
                f"[DomainRandomizer] Service '{client.srv_name}' KHÔNG TỒN "
                f"TẠI/không discover được sau {timeout_sec}s."
            )
            return None

        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            self._node, future, executor=self._executor, timeout_sec=timeout_sec
        )
        elapsed = time.monotonic() - t_start

        if not future.done():
            self._logger.warn(
                f"[DomainRandomizer] Service '{client.srv_name}' TIMEOUT "
                f"sau {elapsed:.2f}s (không nhận được phản hồi)."
            )
            # Huỷ future còn treo để tránh rò rỉ.
            future.cancel()
            return None

        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001 - log mọi lỗi service
            self._logger.warn(
                f"[DomainRandomizer] Service '{client.srv_name}' lỗi sau "
                f"{elapsed:.2f}s: {exc}"
            )
            return None

        self._logger.info(
            f"[DomainRandomizer] Service '{client.srv_name}' phản hồi sau "
            f"{elapsed:.2f}s: success={getattr(resp, 'success', None)}"
        )
        return resp

    def _delete_entity(self) -> bool:
        req = DeleteEntity.Request()
        req.entity = Entity()
        req.entity.name = self._model_name
        req.entity.type = Entity.MODEL
        resp = self._call_service_sync(self._delete_client, req, self._timeout)
        return bool(resp is not None and resp.success)

    def _spawn_entity(self, sdf: str, pose: Pose) -> bool:
        req = SpawnEntity.Request()
        req.entity_factory.name = self._model_name
        req.entity_factory.sdf = sdf
        req.entity_factory.pose = pose
        req.entity_factory.relative_to = "world"
        req.entity_factory.allow_renaming = False
        resp = self._call_service_sync(self._spawn_client, req, self._timeout)
        return bool(resp is not None and resp.success)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Dọn dẹp node + executor riêng. Gọi trong `env.close()`."""
        try:
            self._executor.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._node.destroy_node()
        except Exception:  # noqa: BLE001
            pass


#  WRAPPER GỌI DomainRandomizer bên trg reset()
class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        randomizer: Optional[DomainRandomizer] = None,
        spawn_pose: Optional[Pose] = None,
        **randomizer_kwargs: Any,
    ):
        super().__init__(env)
        self._last_physics: Optional[RandomizedPhysics] = None
        self._owns_randomizer = randomizer is None

        if spawn_pose is not None:
            self._spawn_pose = spawn_pose
        else:
            self._spawn_pose = Pose()
            # Giá trị mặc định lấy từ pose ban đầu của target_box trong
            # pick_and_place_world.sdf (0.5, 0.3, 0.83) - chỉ cần hợp lệ,
            # sẽ bị env.reset() gốc ghi đè vị trí ngay sau đó.
            self._spawn_pose.position.x = 0.5
            self._spawn_pose.position.y = 0.3
            self._spawn_pose.position.z = 0.83
            self._spawn_pose.orientation.w = 1.0

        self._randomizer = randomizer or DomainRandomizer(**randomizer_kwargs)

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self._last_physics = self._randomizer.randomize(self._spawn_pose)
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        if self._last_physics is not None:
            info["domain_randomization"] = {
                "mass": self._last_physics.mass,
                "mu1": self._last_physics.mu1,
                "mu2": self._last_physics.mu2,
            }
        return obs, info

    def close(self):
        # Dọn dẹp node/executor riêng của randomizer TRƯỚC khi đóng env
        # gốc (thứ tự ngược lại lúc khởi tạo).
        if self._owns_randomizer:
            self._randomizer.close()
        return self.env.close()

    @property
    def last_physics(self) -> Optional[RandomizedPhysics]:
        """Tham số vật lý random ở episode gần nhất - dùng để log vào
        TensorBoard/CSV trong `train_sac.py` (qua 1 callback tuỳ biến)."""
        return self._last_physics
