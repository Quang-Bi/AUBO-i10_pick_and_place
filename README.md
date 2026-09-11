# AUBO-i10_pick_and_place
Điều khiển robot AUBO-i10 bằng Reinforcement Learning định hướng Sim2Real.

I. GIỚI THIỆU:
- Pipeline Reinforcement Learning (SAC) trên ROS 2 Jazzy + Gazebo Harmonic, điều khiển cánh tay robot AUBO-i10 kèm gripper DH-AG95 thực hiện tác vụ Pick-and-Place, được thiết kế hướng tới chuyển giao Sim2Real.
- Tài liệu này sẽ hướng dẫn cài đặt, khởi tạo training cho robot.
- Robot đến thời điểm hiện tại mới đang training ở giải đoạn 1 (gắp hộp với vị trí cố định), với success_rate = 0. Dự success_rate lên tới 60% sẽ cho hộp spawn ở vị trí lân cận ngẫu nhiên, và sau đó khi đạt success_rate 70%, domain randomization sẽ bắt đầu được áp dụng.


II. DEPENDENCIES:
- ROS2 Jazzy
- Gazebo Harmonic
- Các phần phụ thuộc python có thể được cài đặt từ tệp require.txt
```shell
pip install --break-system-packages -r requirements.txt
```

III. Chạy training:

Load AUBO-i10, gripper DH-AG95, pick_and_place_world lên Gazebo, kích hoạt controllers
```shell
ros2 launch auboi10_bringup auboi10_launch.py
```

Tiếp tục training từ checkpoint 90000, nếu muốn áp dụng randomization thì tắt cờ disable domain_randomization
```shell
ros2 run gymnasium_auboi10 train_sac --resume-from ~/runs/20260909_123359/checkpoints/sac_aubo_pick_place_90000_steps --disable-domain-randomization
```

Nếu muốn training với target box được spawn ngẫu nhiên vị trí thì đọc chú thích ở dòng 295, 296 trong file aubo_env.py