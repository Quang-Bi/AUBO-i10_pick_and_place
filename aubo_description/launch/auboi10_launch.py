
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch.substitutions import Command
from launch_ros.actions import Node


def generate_launch_description():
    # ------------------------------------------------------------
    # Đường dẫn - GIỮ NGUYÊN như trong lệnh terminal gốc của bạn.
    # Nếu sau này package aubo_description được cài đặt đúng chuẩn
    # (install(DIRECTORY world/urdf ...) trong CMakeLists.txt), nên
    # đổi sang FindPackageShare('aubo_description') để không phụ
    # thuộc đường dẫn tuyệt đối trong thư mục src/.
    # ------------------------------------------------------------
    ws_src = os.path.expanduser("~/ros2_ws/src")
    aubo_description_dir = os.path.join(ws_src, "aubo_description")
    world_file = os.path.join(aubo_description_dir, "pick_and_place_world.sdf")
    xacro_file = os.path.join(aubo_description_dir, "urdf", "aubo_i10_dh_ag95.xacro")

    # ------------------------------------------------------------
    # Biến môi trường (Terminal 1) - PHẢI set TRƯỚC khi tiến trình
    # Gazebo được tạo ra, nên đặt ở đầu LaunchDescription.
    # ------------------------------------------------------------
    existing_plugin_path = os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
    new_plugin_path = (
        f"{existing_plugin_path}:/opt/ros/jazzy/lib"
        if existing_plugin_path
        else "/opt/ros/jazzy/lib"
    )
    ament_prefix_path = os.environ.get("AMENT_PREFIX_PATH", "")

    set_plugin_path = SetEnvironmentVariable(
        name="GZ_SIM_SYSTEM_PLUGIN_PATH", value=new_plugin_path
    )
    set_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH", value=ament_prefix_path
    )

    # ------------------------------------------------------------
    # Terminal 1: Gazebo Harmonic
    # `gz sim` không phải node ROS 2 (không dùng `ros2 run`), nên
    # dùng ExecuteProcess thay vì Node. `cwd=` thay cho lệnh `cd`.
    # ------------------------------------------------------------
    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "r", world_file],
        cwd=aubo_description_dir,
        output="screen",
    )

    # ------------------------------------------------------------
    # Terminal 2: robot_state_publisher
    # `Command([...])` chạy `xacro <file>` lúc launch khởi động và
    # dùng kết quả (chuỗi URDF) làm giá trị tham số robot_description
    # - tương đương chính xác `$(xacro ...)` trong lệnh bash gốc.
    # ------------------------------------------------------------
    robot_description = Command(["xacro ", xacro_file])

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    # ------------------------------------------------------------
    # Terminal 3: Spawn robot vào Gazebo (trễ 4s so với lúc bắt đầu
    # để Gazebo có thời gian load world xong trước khi spawn).
    # ------------------------------------------------------------
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-topic", "robot_description", "-name", "aubo_i10", "-z", "0.1"],
        output="screen",
    )

    # ------------------------------------------------------------
    # Terminal 4: 3 controller (trễ thêm sau spawn, để chắc chắn
    # robot + plugin gz_ros2_control đã khởi tạo xong controller_manager).
    # ------------------------------------------------------------
    spawn_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )
    spawn_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
        output="screen",
    )
    spawn_gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller"],
        output="screen",
    )

    # ------------------------------------------------------------
    # Terminal 5: ros_gz_bridge - có thể chạy sớm cùng lúc với Gazebo,
    # bridge tự chờ/khám phá lại service khi world đã sẵn sàng, không
    # cần trễ thêm.
    # ------------------------------------------------------------
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/model/target_box/pose@geometry_msgs/msg/Pose[gz.msgs.Pose",
            "/world/pick_and_place_world/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            "/world/pick_and_place_world/remove@ros_gz_interfaces/srv/DeleteEntity",
            "/world/pick_and_place_world/create@ros_gz_interfaces/srv/SpawnEntity",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            set_plugin_path,
            set_resource_path,
            # Khởi động ngay: Gazebo, robot_state_publisher, bridge.
            gazebo,
            robot_state_publisher,
            bridge,
            # Trễ 4s: spawn robot (đợi world load xong).
            TimerAction(period=4.0, actions=[spawn_robot]),
            # Trễ 7s: 3 controller (đợi robot spawn + ros2_control
            # plugin khởi tạo controller_manager xong).
            TimerAction(
                period=7.0,
                actions=[
                    spawn_joint_state_broadcaster,
                    spawn_arm_controller,
                    spawn_gripper_controller,
                ],
            ),
        ]
    )
