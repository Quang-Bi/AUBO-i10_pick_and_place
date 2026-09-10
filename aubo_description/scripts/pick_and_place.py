#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time

class AuboPickAndPlace(Node):
    def __init__(self):
        super().__init__('aubo_pick_and_place_node')
        
        # Action client cho cánh tay Aubo i10 (Dùng FollowJointTrajectory)
        self.arm_action_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
            
        # Action client cho tay gắp DH AG95 (Dùng GripperCommand)
        # Lưu ý: Tên action server thường là '/gripper_controller/gripper_cmd', 
        # hãy kiểm tra lại bằng lệnh 'ros2 action list' nếu cần.
        self.gripper_action_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd')

        # Danh sách các khớp cánh tay dựa trên file URDF[cite: 1, 4]
        self.arm_joint_names = [
            'shoulder_joint', 
            'upperArm_joint', 
            'foreArm_joint', 
            'wrist1_joint', 
            'wrist2_joint', 
            'wrist3_joint'
        ]

        self.get_logger().info('Đang chờ các Action Server...')
        self.arm_action_client.wait_for_server()
        self.gripper_action_client.wait_for_server()
        self.get_logger().info('Action Servers đã sẵn sàng!')

    def send_arm_goal(self, joint_angles, duration_sec):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.arm_joint_names
        
        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start = Duration(sec=duration_sec, nanosec=0)
        
        goal_msg.trajectory.points.append(point)
        
        self.get_logger().info(f'Đang di chuyển cánh tay tới: {joint_angles}')
        return self.arm_action_client.send_goal_async(goal_msg)

    def send_gripper_goal(self, position):
        # Giới hạn của tay gắp là từ 0.0 đến 0.93[cite: 3]
        clamped_pos = max(0.0, min(position, 0.93))
        
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = clamped_pos
        # Lực gắp tối đa theo file cấu hình là 50.0[cite: 3]
        goal_msg.command.max_effort = 50.0 
        
        self.get_logger().info(f'Đang điều khiển tay gắp tới vị trí: {clamped_pos}')
        return self.gripper_action_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AuboPickAndPlace()

    home_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pre_pick_pose = [0.5, -0.5, 1.0, -0.5, 1.57, 0.0]
    pick_pose = [0, -0.7, 1.2, -0.5, 1.57, 0.0]
    pre_place_pose = [-0.5, -0.5, 1.0, -0.5, 1.57, 0.0]
    place_pose = [-0.5, -0.7, 1.2, -0.5, 1.57, 0.0]
      
    # Thực thi tuần tự
    node.send_gripper_goal(0.0) # Mở tay gắp
    time.sleep(2.0)

    node.send_arm_goal(pre_pick_pose, 4)
    time.sleep(4.5)

    node.send_arm_goal(pick_pose, 2)
    time.sleep(2.5)

    node.send_gripper_goal(0.65) # Đóng tay gắp để kẹp vật
    time.sleep(2.0)

    node.send_arm_goal(pre_pick_pose, 2)
    time.sleep(2.5)

    node.send_arm_goal(pre_place_pose, 4)
    time.sleep(4.5)

    node.send_arm_goal(place_pose, 2)
    time.sleep(2.5)

    node.send_gripper_goal(0.0) # Mở tay gắp để thả vật
    time.sleep(2.0)

    node.send_arm_goal(home_pose, 4)
    time.sleep(4.5)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()