from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_io import RTDEIOInterface as RTDEIo
import time
import math

class Pose():
    def __init__(self, x, y, z, rx, ry, rz) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_z(self, z):
        self.z = z

    def set_rx(self, rx):
        self.rx = rx

    def set_ry(self, ry):
        self.ry = ry

    def set_rz(self, rz):
        self.rz = rz

    def get_pose(self):
        return [
            self.x,
            self.y,
            self.z,
            self.rx,
            self.ry,
            self.rz
        ]
    
def targetPoseReached(curr_pose:Pose, goal_pose:Pose, epsilon):
    return math.sqrt((goal_pose.x-curr_pose.x)**2 + (goal_pose.y-curr_pose.y)**2 + (goal_pose.z-curr_pose.z)**2) < epsilon


class RobotController():
    def __init__(self, ip_address, id_open, id_close) -> None:
        self.ip_address = ip_address
        self.rtde_frequency = 500.0
        self.rtde_c = RTDEControl(self.ip_address, self.rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
        # self.rtde_c = RTDEControl(self.ip_address)
        self.rtde_r = RTDEReceive(self.ip_address)
        self.rtde_io = RTDEIo(self.ip_address)
        self.openGripperDO = id_open
        self.closeGripperDO = id_close

    def __del__(self):
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        self.rtde_io.disconnect()

    def send_pose(self, pose:Pose, epsilon=0.001, verbose=False):
        if self.rtde_c.isConnected():
            self.rtde_c.moveL(pose)
            while not targetPoseReached(Pose(*self.get_tcp_pose()), Pose(*pose), epsilon):
                pass
            if verbose:
                print("Goal pose reached")

    # def isTargetPoseReached(self, target_pose, epsilon):
    #     curr_pose = self.get_tcp_pose()



    def send_joints(self, joint_angles, verbose=False):
        if self.rtde_c.isConnected():
            self.rtde_c.moveJ(joint_angles)
            while not self.rtde_c.isSteady():
                pass
            if verbose:
                print("Goal joints reached")

    def get_tcp_pose(self):
        if self.rtde_r.isConnected():
            return self.rtde_r.getActualTCPPose()
        return None

    def get_joint_positions(self):
        if self.rtde_r.isConnected():
            return self.rtde_r.getActualQ()
        return None

    def open_gripper(self, verbose=False):
        if not self.rtde_r.getDigitalOutState(self.openGripperDO):
            self.rtde_io.setStandardDigitalOut(self.openGripperDO, True)
            time.sleep(0.01)
            while self.rtde_r.getDigitalOutState(self.openGripperDO):
                pass
            if verbose:
                print("Gripper opened.")

    def close_gripper(self, verbose=False):
        if not self.rtde_r.getDigitalOutState(self.closeGripperDO):
            self.rtde_io.setStandardDigitalOut(self.closeGripperDO, True)
            time.sleep(0.01)
            while self.rtde_r.getDigitalOutState(self.closeGripperDO):
                pass
            if verbose:
                print("Gripper closed.")
            


# if __name__ == "__main__":

#     UR5e = RobotController("172.16.0.2", 0, 1)
#     UR5e.close_gripper()
#     UR5e.open_gripper()

#     curr_pose = UR5e.get_tcp_pose()
#     print(f'Current pose: {curr_pose}')
#     next_pose = curr_pose.copy()
#     next_pose[2] = next_pose[2] - 0.1
#     print(f'Next pose: {next_pose}')
#     UR5e.send_pose(next_pose)

#     curr_joints = UR5e.get_joint_positions()
#     print(f'Current joints: {curr_joints}')