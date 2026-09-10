
import argparse
import sys
import time
import traceback

import numpy as np

from gymnasium_auboi10.aubo_env import AuboPickAndPlaceEnv


def _check_finite(name: str, arr: np.ndarray) -> None:
    """Kiểm tra không có NaN/Inf để phát hiện sớm lỗi số học / mất dữ liệu ROS."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"[test_env] '{name}' chứa giá trị NaN/Inf: {arr}")


def run_random_rollout(episodes: int, steps_per_episode: int, control_hz: float) -> int:
    env = AuboPickAndPlaceEnv(
        node_name="aubo_pick_place_env_test",
        control_hz=control_hz,
        max_episode_steps=steps_per_episode,
    )

    exit_code = 0
    try:
        # ---- Kiểm tra cơ bản về spaces trước khi rollout ----
        print(f"[test_env] action_space      = {env.action_space}")
        print(f"[test_env] observation_space = {env.observation_space}")
        assert env.action_space.shape == (7,), "Action space phải có 7 chiều (6 tay + 1 kẹp)."

        for ep in range(episodes):
            print(f"\n[test_env] ===== Episode {ep + 1}/{episodes} =====")
            t0 = time.time()
            obs, info = env.reset(seed=ep)

            # Kiểm tra shape/dtype/finite của observation ban đầu.
            assert obs.shape == env.observation_space.shape, (
                f"Obs shape {obs.shape} != observation_space.shape "
                f"{env.observation_space.shape}"
            )
            _check_finite("reset().obs", obs)
            print(f"[test_env] reset() OK trong {time.time() - t0:.2f}s, obs[:6] (q)={obs[:6]}")

            ep_return = 0.0
            for t in range(steps_per_episode):
                action = env.action_space.sample()

                step_t0 = time.time()
                obs, reward, terminated, truncated, info = env.step(action)
                step_dt = time.time() - step_t0

                # ---- Các kiểm tra "chống deadlock / chống lỗi shape" ----
                assert obs.shape == env.observation_space.shape, (
                    f"[step {t}] Obs shape sai: {obs.shape}"
                )
                assert np.isscalar(reward) or isinstance(reward, (int, float)), (
                    f"[step {t}] reward phải là số vô hướng, nhận được: {type(reward)}"
                )
                assert isinstance(terminated, (bool, np.bool_)), (
                    f"[step {t}] terminated phải là bool, nhận được {type(terminated)}"
                )
                assert isinstance(truncated, (bool, np.bool_)), (
                    f"[step {t}] truncated phải là bool, nhận được {type(truncated)}"
                )
                _check_finite(f"step({t}).obs", obs)
                _check_finite(f"step({t}).reward", np.array([reward]))

                if step_dt > 3.0 * (1.0 / control_hz):
                    print(
                        f"[test_env][CẢNH BÁO] step {t} mất {step_dt:.3f}s, "
                        f"lâu hơn nhiều so với chu kỳ điều khiển kỳ vọng "
                        f"({1.0 / control_hz:.3f}s). Có thể do action/service "
                        "ROS 2 đang bị chặn (khả năng deadlock)."
                    )

                ep_return += float(reward)

                if t % 10 == 0:
                    print(
                        f"[test_env] step={t:03d} reward={reward:+.3f} "
                        f"dist_ee_obj={info.get('dist_ee_obj', float('nan')):.3f} "
                        f"dist_obj_target={info.get('dist_obj_target', float('nan')):.3f} "
                        f"terminated={terminated} truncated={truncated}"
                    )

                if terminated or truncated:
                    print(
                        f"[test_env] Kết thúc episode ở step {t} "
                        f"(terminated={terminated}, truncated={truncated}, "
                        f"is_success={info.get('is_success')})"
                    )
                    break

            print(f"[test_env] Tổng reward episode {ep + 1}: {ep_return:.3f}")

        print("\n[test_env]  Random rollout hoàn tất KHÔNG có lỗi shape/NaN/deadlock rõ ràng.")

    except AssertionError as exc:
        print(f"\n[test_env] ❌ THẤT BẠI (assertion): {exc}")
        traceback.print_exc()
        exit_code = 1
    except Exception as exc:  # noqa: BLE001 - test script: muốn bắt mọi lỗi để báo cáo rõ ràng
        print(f"\n[test_env] ❌ LỖI KHÔNG MONG MUỐN: {exc}")
        traceback.print_exc()
        exit_code = 1
    finally:
        # Luôn đóng node/executor để không treo tiến trình (nguồn deadlock phổ biến).
        print("[test_env] Đang đóng môi trường (rclpy shutdown)...")
        env.close()

    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Random rollout smoke test cho AuboPickAndPlaceEnv."
    )
    parser.add_argument("--episodes", type=int, default=3, help="Số episode để chạy thử.")
    parser.add_argument(
        "--steps", type=int, default=50, help="Số bước tối đa mỗi episode."
    )
    parser.add_argument(
        "--control-hz", type=float, default=20.0, help="Tần số điều khiển (Hz)."
    )
    args = parser.parse_args()

    exit_code = run_random_rollout(
        episodes=args.episodes,
        steps_per_episode=args.steps,
        control_hz=args.control_hz,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
