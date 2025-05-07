from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_io import RTDEIOInterface as RTDEIo
import time


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

    def send_pose(self, pose):
        if self.rtde_c.isConnected():
            self.rtde_c.moveL(pose)

    def send_joints(self, joint_angles):
        if self.rtde_c.isConnected():
            self.rtde_c.moveJ(joint_angles)

    def get_tcp_pose(self):
        if self.rtde_r.isConnected():
            return self.rtde_r.getActualTCPPose()
        return None

    def get_joint_positions(self):
        if self.rtde_r.isConnected():
            return self.rtde_r.getActualQ()
        return None

    def open_gripper(self, verbose=True):
        if not self.rtde_r.getDigitalOutState(self.openGripperDO):
            self.rtde_io.setStandardDigitalOut(self.openGripperDO, True)
            time.sleep(0.01)
            while self.rtde_r.getDigitalOutState(self.openGripperDO):
                pass
            if verbose:
                print("Gripper opened.")

    def close_gripper(self, verbose=True):
        if not self.rtde_r.getDigitalOutState(self.closeGripperDO):
            self.rtde_io.setStandardDigitalOut(self.closeGripperDO, True)
            time.sleep(0.01)
            while self.rtde_r.getDigitalOutState(self.closeGripperDO):
                pass
            if verbose:
                print("Gripper closed.")
            


