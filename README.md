# AUBO-i10_pick_and_place
Điều khiển robot AUBO-i10 bằng Reinforcement Learning định hướng Sim2Real.

I. GIỚI THIỆU:
- Pipeline Reinforcement Learning (SAC) trên ROS 2 Jazzy + Gazebo Harmonic, điều khiển cánh tay robot Aubo i10 (6 bậc tự do) kèm gripper DH-AG95 thực hiện tác vụ Pick-and-Place, được thiết kế hướng tới chuyển giao Sim2Real.
- Tài liệu này sẽ hướng dẫn cài đặt, chạy và reproduce hệ thống.
- Robot đến thời điểm hiện tại mới đang training ở giải đoạn 1 (gắp hộp với vị trí cố định). Dự success_rate lên tới 60% sẽ cho hộp spawn ở vị trí lân
  cận ngẫu nhiên, và sau đó khi đạt success_rate 80%, domain randomization sẽ bắt đầu được áp dụng.

  Nếu muốn bỏ qua giai đoạn một

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

Start MoveIt for motion planning
```shell
roslaunch panda_sim_moveit sim_move_group.launch
```

Run the object detector
```shell
rosrun pick_and_place object_detector.py
```

Run the pick-and-place controller
```shell
rosrun pick_and_place pick_and_place_state_machine.py
```
