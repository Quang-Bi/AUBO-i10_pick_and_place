
import argparse
import csv
import sys

import numpy as np
import rclpy
import matplotlib

matplotlib.use("Agg")  # không cần màn hình hiển thị, chỉ xuất file PNG
import matplotlib.pyplot as plt

from stable_baselines3 import SAC

try:
    from .aubo_env import AuboPickAndPlaceEnv, ARM_JOINT_NAMES, ARM_JOINT_VEL_LIMIT, ARM_JOINT_ACCEL_LIMIT
except ImportError:  # pragma: no cover - fallback khi chạy như script rời
    from aubo_env import AuboPickAndPlaceEnv, ARM_JOINT_NAMES, ARM_JOINT_VEL_LIMIT, ARM_JOINT_ACCEL_LIMIT


def run_episode_and_log(model_path: str, max_steps: int, deterministic: bool):
    rclpy.init()
    env = AuboPickAndPlaceEnv(node_name="aubo_trajectory_logger")
    model = SAC.load(model_path, env=None, device="cpu")

    n_arm = len(ARM_JOINT_NAMES)
    records = []
    prev_qdot_cmd = np.zeros(n_arm, dtype=np.float64)

    try:
        obs, info = env.reset()
        t = 0.0
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            # q_cmd/qdot_cmd: đầu ra THỰC SỰ của _JointVelAccelLimiter ở bước này.
            q_cmd = np.asarray(env._q_cmd, dtype=np.float64).copy()
            qdot_cmd = np.asarray(env._qdot_cmd, dtype=np.float64).copy()
            qddot_cmd = (qdot_cmd - prev_qdot_cmd) / env.control_period
            prev_qdot_cmd = qdot_cmd.copy()

            # q/qdot: giá trị ĐO ĐƯỢC thực tế, ghi thêm để đối chiếu bám quỹ đạo.
            q_meas, qdot_meas = env._node.get_arm_state()

            row = {"t": t, "step": step, "reward": reward, "safety_violation": info.get("safety_violation")}
            for i, jname in enumerate(ARM_JOINT_NAMES):
                row[f"q_cmd_{jname}"] = q_cmd[i]
                row[f"qdot_cmd_{jname}"] = qdot_cmd[i]
                row[f"qddot_cmd_{jname}"] = qddot_cmd[i]
                row[f"q_meas_{jname}"] = float(q_meas[i])
                row[f"qdot_meas_{jname}"] = float(qdot_meas[i])
            records.append(row)

            t += env.control_period
            if terminated or truncated:
                print(f"Episode kết thúc ở bước {step} (terminated={terminated}, truncated={truncated}).")
                break
    finally:
        env.close()

    return records


def write_csv(records, path: str):
    if not records:
        print("Không có dữ liệu để ghi CSV (episode kết thúc ngay bước 0?).")
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Đã ghi log chi tiết: {path}")


def plot_trajectory(records, output_path: str):
    if not records:
        return
    t = np.array([r["t"] for r in records])
    n_arm = len(ARM_JOINT_NAMES)

    q_cmd = np.array([[r[f"q_cmd_{j}"] for j in ARM_JOINT_NAMES] for r in records])
    qdot_cmd = np.array([[r[f"qdot_cmd_{j}"] for j in ARM_JOINT_NAMES] for r in records])
    qddot_cmd = np.array([[r[f"qddot_cmd_{j}"] for j in ARM_JOINT_NAMES] for r in records])

    vel_limit_tightest = min(ARM_JOINT_VEL_LIMIT.values())
    accel_limit_tightest = min(ARM_JOINT_ACCEL_LIMIT.values())

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)

    for i, jname in enumerate(ARM_JOINT_NAMES):
        axes[0].plot(t, q_cmd[:, i], label=jname)
        axes[1].plot(t, qdot_cmd[:, i], label=jname)
        axes[2].plot(t, qddot_cmd[:, i], label=jname)

    axes[0].set_ylabel("Vị trí khớp q_cmd (rad)")
    axes[0].set_title("Quỹ đạo vị trí khớp đã ra lệnh (q_cmd) theo thời gian")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(vel_limit_tightest, color="red", linestyle="--", linewidth=1, label="Giới hạn vận tốc chặt nhất")
    axes[1].axhline(-vel_limit_tightest, color="red", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Vận tốc khớp qdot_cmd (rad/s)")
    axes[1].set_title("Quỹ đạo vận tốc đã ra lệnh - không bao giờ vượt giới hạn (đường đỏ)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(accel_limit_tightest, color="red", linestyle="--", linewidth=1, label="Giới hạn gia tốc chặt nhất")
    axes[2].axhline(-accel_limit_tightest, color="red", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Gia tốc khớp qddot_cmd (rad/s²)")
    axes[2].set_xlabel("Thời gian (giây)")
    axes[2].set_title("Quỹ đạo gia tốc đã ra lệnh - không bao giờ vượt giới hạn (đường đỏ)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Đã lưu biểu đồ: {output_path}")

    # ---- Bằng chứng định lượng (in kèm ra terminal, chèn thẳng vào báo cáo) ----
    max_abs_vel = np.max(np.abs(qdot_cmd))
    max_abs_accel = np.max(np.abs(qddot_cmd))
    print("\n=== TÓM TẮT ĐỊNH LƯỢNG (chèn vào báo cáo kỹ thuật) ===")
    print(f"Vận tốc |qdot_cmd| lớn nhất quan sát được : {max_abs_vel:.4f} rad/s")
    print(f"Giới hạn vận tốc chặt nhất cho phép        : {vel_limit_tightest:.4f} rad/s")
    print(f"=> {'ĐẠT' if max_abs_vel <= vel_limit_tightest + 1e-6 else 'VI PHẠM'}")
    print(f"Gia tốc |qddot_cmd| lớn nhất quan sát được : {max_abs_accel:.4f} rad/s^2")
    print(f"Giới hạn gia tốc chặt nhất cho phép         : {accel_limit_tightest:.4f} rad/s^2")
    print(f"=> {'ĐẠT' if max_abs_accel <= accel_limit_tightest + 1e-6 else 'VI PHẠM'}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Chạy 1 episode demo và ghi log/vẽ biểu đồ quỹ đạo vận tốc-gia tốc."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-prefix", type=str, default="trajectory_demo")
    parser.add_argument("--stochastic", action="store_true")
    args, _ = parser.parse_known_args(argv)

    records = run_episode_and_log(
        model_path=args.model_path,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
    )
    write_csv(records, f"{args.output_prefix}.csv")
    plot_trajectory(records, f"{args.output_prefix}.png")

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
