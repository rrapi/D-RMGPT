from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl


class RobotController():
    def __init__(self, ip_address) -> None:
        self.ip_address = ip_address
        self.rtde_c = RTDEControl(self.ip_address)
        self.rtde_r = RTDEReceive(self.ip_address)

    def __del__(self):
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        self.rtde_io.disconnect()

    def send_pose(self, pose):
        if self.rtde_c.isConnected():
            self.rtde_c.moveL(pose)
            
    def send_path(self, path):
        if self.rtde_c.isConnected():
            self.rtde_c.moveL(path)

    def get_tcp_pose(self):
        if self.rtde_r.isConnected():
            return self.rtde_r.getActualTCPPose()
        return None



