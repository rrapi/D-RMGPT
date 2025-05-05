import socket
import time

# UR5e robot IP address (change to match your setup)
UR5_IP = "172.16.0.2"
PORT = 30002  # UR Secondary Interface

# Connect to the robot
def connect_to_robot(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    print(f"Connected to UR5e at {ip}:{port}")
    return s

# Send URScript command
def send_command(sock, command):
    sock.sendall(command.encode('utf-8'))
    time.sleep(0.1)  # Allow time for the robot to process
    print(f"Sent command: {command}")

# RG6 gripper commands via OnRobot URCap
def activate_gripper(sock):
    cmd = 'rg_activate_and_wait()\n'
    send_command(sock, cmd)

def open_gripper(sock):
    cmd = 'rg_open_and_wait()\n'
    send_command(sock, cmd)

def close_gripper(sock):
    cmd = 'rg_close_and_wait()\n'
    send_command(sock, cmd)

def move_robot(sock, pose, a=1.2, v=0.25):
    cmd = f'movel({pose}, {a}, {v})'
    send_command(sock, cmd)

# Set gripper position, force, and speed (0–255 scale)
def move_gripper(sock, position=255, force=100, speed=100):
    cmd = f'rg_move_and_wait({position}, {force}, {speed})\n'
    send_command(sock, cmd)

# Main program
if __name__ == "__main__":
    try:
        sock = connect_to_robot(UR5_IP, PORT)
        
        # Example usage:
        activate_gripper(sock)
        time.sleep(2)

        move_gripper(sock, position=0)  # Fully open
        time.sleep(2)

        move_gripper(sock, position=255)  # Fully close
        time.sleep(2)

        move_robot(sock, pose = [0.2, 0.4, 0.3, 0, 3.14, 0])
        time.sleep(2)

        open_gripper(sock)
        time.sleep(2)

        close_gripper(sock)
        time.sleep(2)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        print("Connection closed.")
