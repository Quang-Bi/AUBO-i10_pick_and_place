import argparse
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import rclpy

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv

# Import tương đối (khi chạy như 1 module trong package ROS 2) với
# fallback sang import tuyệt đối (khi chạy trực tiếp `python3 train_sac.py`
# từ trong thư mục package, ví dụ lúc debug nhanh).
try:
    from .aubo_env import AuboPickAndPlaceEnv
    from .sim2real_wrappers import SensorNoiseWrapper, ActionLatencyWrapper
    from .domain_randomizer import DomainRandomizationWrapper
except ImportError:  # pragma: no cover - fallback khi chạy như script rời
    from aubo_env import AuboPickAndPlaceEnv
    from sim2real_wrappers import SensorNoiseWrapper, ActionLatencyWrapper
    from domain_randomizer import DomainRandomizationWrapper


# =====================================================================
# Callback: lưu best_model.zip dựa trên reward huấn luyện (rolling mean)
# =====================================================================
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """Định kỳ đọc log CSV của `Monitor`, tính trung bình động của
    reward/episode trên `check_freq` bước gần nhất, và lưu
    `best_model.zip` mỗi khi giá trị này CAO HƠN kỷ lục trước đó.

    Đây là cách tiếp cận được khuyến nghị trong tài liệu SB3 khi không
    có `eval_env` riêng (chỉ có 1 instance Gazebo/robot thật)."""

    def __init__(
        self,
        check_freq: int,
        log_dir: str,
        rolling_window: int = 20,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.rolling_window = rolling_window
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        try:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
        except Exception:  # noqa: BLE001 - chưa đủ dữ liệu log ở early-run
            return True

        if len(x) == 0:
            return True

        mean_reward = float(np.mean(y[-self.rolling_window:]))
        if self.verbose > 0:
            print(
                f"[SaveOnBestTrainingRewardCallback] step={self.num_timesteps} "
                f"reward_TB_{self.rolling_window}ep={mean_reward:.2f} "
                f"(best={self.best_mean_reward:.2f})"
            )

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print(f"  -> Reward cải thiện, lưu model tại {self.save_path}.zip")
            self.model.save(self.save_path)

        return True



# Callback: log tham số Domain Randomization (mass/mu1/mu2) mỗi episode
class DomainRandomizationLoggingCallback(BaseCallback):
    """Ghi mass/mu1/mu2 vừa random hoá (nếu có) vào SB3 logger
    (-> hiển thị trên TensorBoard cùng các đại lượng huấn luyện khác)."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            dr_info: Optional[Dict[str, Any]] = info.get("domain_randomization")
            if dr_info is not None:
                self.logger.record("domain_rand/mass", dr_info["mass"])
                self.logger.record("domain_rand/mu1", dr_info["mu1"])
                self.logger.record("domain_rand/mu2", dr_info["mu2"])
        return True

# Callback: Curriculum thu nhỏ/nới rộng dần biên độ hành động (max_delta_frac)
class SafetyCurriculumCallback(BaseCallback):
    """Tăng dần `max_delta_frac` của môi trường từ `start_frac` -> `end_frac`
    tuyến tính theo số bước huấn luyện đã đi qua (`num_timesteps`), thay vì
    dùng ngay biên độ hành động tối đa từ đầu.

    Lý do: ở giai đoạn đầu huấn luyện (policy gần như ngẫu nhiên, đặc
    biệt trước khi `learning_starts`), hành động có biên độ lớn dễ gây
    chuyển động hỗn loạn -> tự va đập / đập bàn (đúng vấn đề quan sát
    được trong thực tế). Bắt đầu với biên độ nhỏ giúp giới hạn "mức độ
    nguy hiểm" của các hành động ngẫu nhiên ban đầu, và nới rộng dần khi
    policy đã bắt đầu học được các chuyển động có ý nghĩa.

    Gọi `env.set_max_delta_frac(...)` xuyên qua toàn bộ chuỗi wrapper
    nhờ cơ chế `__getattr__` mặc định của `gymnasium.Wrapper` (tự động
    chuyển tiếp xuống `AuboPickAndPlaceEnv` ở lớp trong cùng)."""

    def __init__(
        self,
        start_frac: float = 0.15,
        end_frac: float = 0.60,
        ramp_timesteps: int = 100_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.start_frac = start_frac
        self.end_frac = end_frac
        self.ramp_timesteps = max(1, ramp_timesteps)
        self._last_logged_frac: Optional[float] = None

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.ramp_timesteps)
        frac = self.start_frac + progress * (self.end_frac - self.start_frac)

        # Chỉ gọi env_method (và in log) khi giá trị thực sự đổi đáng kể,
        # tránh gọi xuyên suốt wrapper chain ở MỌI bước gây overhead thừa.
        if self._last_logged_frac is None or abs(frac - self._last_logged_frac) > 1e-3:
            self.training_env.env_method("set_max_delta_frac", frac)
            self._last_logged_frac = frac
            if self.verbose > 0 and self.num_timesteps % 5000 == 0:
                print(
                    f"[SafetyCurriculumCallback] step={self.num_timesteps} "
                    f"max_delta_frac={frac:.3f} (progress={progress:.2%})"
                )

        self.logger.record("safety/max_delta_frac", frac)
        return True

# Callback: Giám sát tỉ lệ vi phạm an toàn (va bàn / kẹt-tự va đập)
class SafetyMonitoringCallback(BaseCallback):
    """Đếm & log tỉ lệ episode kết thúc do vi phạm an toàn
    (`info["safety_violation"]` từ Safety Watchdog trong `aubo_env.py`),
    tách riêng theo loại ("table_collision" / "stall_or_collision"), để
    theo dõi trên TensorBoard xem cơ chế an toàn có đang hoạt động và xu
    hướng vi phạm có giảm dần theo thời gian huấn luyện hay không."""

    def __init__(self, log_every: int = 2000, verbose: int = 1):
        super().__init__(verbose)
        self.log_every = log_every
        self._counts: Dict[str, int] = {"table_collision": 0, "stall_or_collision": 0}
        self._episodes_seen = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))
        for info, done in zip(infos, dones):
            if not done:
                continue
            self._episodes_seen += 1
            violation = info.get("safety_violation")
            if violation in self._counts:
                self._counts[violation] += 1
                if self.verbose > 0:
                    print(
                        f"[SafetyMonitoringCallback] step={self.num_timesteps} "
                        f"VI PHẠM AN TOÀN: {violation}"
                    )

        if self.num_timesteps % self.log_every == 0 and self._episodes_seen > 0:
            for name, count in self._counts.items():
                rate = count / max(1, self._episodes_seen)
                self.logger.record(f"safety/{name}_rate", rate)
                self.logger.record(f"safety/{name}_count", count)

        return True

# Xây dựng môi trường huấn luyện (toàn bộ chuỗi wrapper Sim2Real)
def build_env(args: argparse.Namespace, monitor_path: str):
    env = AuboPickAndPlaceEnv(
        node_name="aubo_pick_place_train_env",
        control_hz=args.control_hz,
        max_episode_steps=args.max_episode_steps,
    )

    # 1) Domain randomization (mass/friction của target_box) - chạy TRƯỚC
    #    tiên trong chuỗi wrapper vì nó tương tác trực tiếp với node ROS
    #    của env gốc, không phụ thuộc dữ liệu observation/action đã bị
    #    biến đổi bởi các wrapper khác.
    if not args.disable_domain_randomization:
        env = DomainRandomizationWrapper(
            env,
            mass_range=(args.mass_min, args.mass_max),
            friction_range=(args.friction_min, args.friction_max),
        )

    # 2) Nhiễu cảm biến (q, qdot, pose vật thể...).
    if not args.disable_sensor_noise:
        env = SensorNoiseWrapper(env)

    # 3) Trễ hành động ngẫu nhiên 20-60ms (mặc định).
    if not args.disable_action_latency:
        env = ActionLatencyWrapper(
            env, latency_range_ms=(args.latency_min_ms, args.latency_max_ms)
        )

    # Monitor PHẢI là wrapper NGOÀI CÙNG để ghi lại đúng reward/độ dài
    # episode "thật" (đã tính cả ảnh hưởng của các wrapper Sim2Real ở trên).
    env = Monitor(env, filename=monitor_path, info_keywords=())
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện SAC cho tác vụ Pick-and-Place Aubo i10 + DH AG95."
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--max-episode-steps", type=int, default=300)

    # Domain randomization
    parser.add_argument("--disable-domain-randomization", action="store_true")
    parser.add_argument("--mass-min", type=float, default=0.05)
    parser.add_argument("--mass-max", type=float, default=0.20)
    parser.add_argument("--friction-min", type=float, default=0.4)
    parser.add_argument("--friction-max", type=float, default=1.5)

    # Sensor noise / action latency
    parser.add_argument("--disable-sensor-noise", action="store_true")
    parser.add_argument("--disable-action-latency", action="store_true")
    parser.add_argument("--latency-min-ms", type=float, default=20.0)
    parser.add_argument("--latency-max-ms", type=float, default=60.0)

    # Safety curriculum (thu nhỏ dần biên độ hành động lúc đầu huấn luyện)
    parser.add_argument(
        "--disable-safety-curriculum", action="store_true",
        help="Tắt curriculum, dùng thẳng max_delta_frac cố định của env "
             "(KHÔNG khuyến nghị - dễ gây va đập lúc policy còn ngẫu nhiên).",
    )
    parser.add_argument("--curriculum-start-frac", type=float, default=0.15)
    parser.add_argument("--curriculum-end-frac", type=float, default=0.60)
    parser.add_argument("--curriculum-ramp-timesteps", type=int, default=100_000)

    # SAC hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=1_000)

    # Logging / checkpoint
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--best-model-check-freq", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Đường dẫn .zip của model đã lưu để load lại và tiếp tục huấn luyện.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    monitor_path = os.path.join(args.log_dir, "monitor.csv")
    # rclpy.init() ph đứng trước mọi thao tác tạo Node (bên trong Env).

    rclpy.init()

    env = None
    try:
        env = build_env(args, monitor_path)
        vec_env = DummyVecEnv([lambda: env])  # n_envs=1: chỉ có 1 world Gazebo

        if args.resume_from is not None:
            print(f"Đang load model từ '{args.resume_from}' để tiếp tục huấn luyện...")
            custom_objects = {
            "learning_rate": args.learning_rate,
            "lr_schedule": lambda _: args.learning_rate,
            }
            model = SAC.load(
                args.resume_from,
                env=vec_env,
                custom_objects=custom_objects,

            )
        else:
            model = SAC(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                learning_starts=args.learning_starts,
                train_freq=16,
                gradient_steps=16,
                ent_coef="auto",
                tensorboard_log=args.log_dir,
                seed=args.seed,
                verbose=1,
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=os.path.join(args.log_dir, "checkpoints"),
            name_prefix="sac_aubo_pick_place",
            save_replay_buffer=True,
        )
        best_model_callback = SaveOnBestTrainingRewardCallback(
            check_freq=args.best_model_check_freq,
            log_dir=args.log_dir,
        )
        dr_logging_callback = DomainRandomizationLoggingCallback()
        safety_monitoring_callback = SafetyMonitoringCallback()

        callback_items = [
            checkpoint_callback,
            best_model_callback,
            dr_logging_callback,
            safety_monitoring_callback,
        ]
        if not args.disable_safety_curriculum:
            callback_items.append(
                SafetyCurriculumCallback(
                    start_frac=args.curriculum_start_frac,
                    end_frac=args.curriculum_end_frac,
                    ramp_timesteps=args.curriculum_ramp_timesteps,
                )
            )
        callbacks = CallbackList(callback_items)

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name="SAC",
            reset_num_timesteps=(args.resume_from is None),
        )

        final_path = os.path.join(args.log_dir, "final_model")
        model.save(final_path)
        print(f"Huấn luyện hoàn tất. Model cuối cùng đã lưu tại: {final_path}.zip")

    except KeyboardInterrupt:
        print("\nNhận Ctrl+C - dừng huấn luyện, đang dọn dẹp ROS 2/Gazebo...")

    finally:
        # ĐÓNG env (node + executor + spin thread) TRƯỚC khi shutdown rclpy,
        # theo đúng thứ tự ngược với lúc khởi tạo.
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                print(f"Lỗi khi đóng môi trường: {exc}")
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
