# -*- coding: utf-8 -*-

from ailab_data.setup import *
import constants
import os
import argparse
from openai import OpenAI
import base64
from camera import RealSenseCamera
import matplotlib.pyplot as plt
# import rtde_control
import time
import cv2
# import numpy as np
import constants
# import socket
from robot_controller import RobotController, Pose
try:
    import winsound
except ImportError:
    def playsound(freq, duration):
        os.system('play -n synth %s sin %s' % (duration/1000, freq))
        # os.system('play -nq -t alsa synth {} sine {}'.format(duration/1000, freq))
else:
    def playsound(freq, duration):
        winsound.Beep(freq, duration)


parser = argparse.ArgumentParser(description='Input arguments')
parser.add_argument('data_dir', type=str, help='Directory of experiment data')


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def write_text_file(file_path, stringa):
    with open(file_path, 'w') as file:
        file.write(stringa)

def show_image(img):
    # read the image 
    # img = cv2.imread(image_path)
    # showing the image
    cv2.imshow('RealSense', img)
    # waiting using waitKey method
    cv2.waitKey(0)


def main():

    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = constants.APIKEY
    model_name 			         = constants.MODEL_NAME

    # DATA_DIR                     = 'ailab_data'
    args = parser.parse_args()
    DATA_DIR = args.data_dir

    UR5e = RobotController("172.16.0.2", 0, 1)
    # UR5e.close_gripper()
    UR5e.open_gripper()
    UR5e.send_joints(array_deg_to_rad(home_joints))


    file_name1                  = 'current_step1.png'
    file_name2                  = 'current_step2.png'
    cwd                         = os.getcwd()
    file_path                   = os.path.join(cwd, DATA_DIR, 'Assembly_sequence_images')

    text_query 			            = read_text_file(os.path.join(cwd, DATA_DIR, 'response_assistant.txt'))
    assistant_robot_planner 	    = read_text_file(os.path.join(cwd, DATA_DIR, 'assistant_robot_planner.txt'))
    # Getting the base64 string of components list image
    base64_image_component_list 	= encode_image(os.path.join(cwd, DATA_DIR, 'components_list.jpg'))

    # RealSenseCamera.check_available_devices()

    camera1=RealSenseCamera(
        device_id='109122072393',
        width=640,
        height=480,
        fps = 30
        )
    camera1.connect()

    camera2=RealSenseCamera(
        device_id='043322071223',
        width=640,
        height=480,
        fps = 30
        )
    camera2.connect()

    # delivery_pose_tmp = delivery_pose.copy()
    delivery_pose_tmp = array_cm_to_m(delivery_pose) + array_deg_to_rad(tool_orientation)

    # connection to OpenAI APIs
    client = OpenAI()
    step = 1
    t_start         = time.time()
    brought_objec   = []
    full_set        = list(range(1, len(object_positions)+1)) #[1, 2, 3, 4, 5, 6, 7, 8, 9]
    while True:

        t_start_proc        = time.time()
        text_query          = read_text_file(os.path.join(cwd, DATA_DIR, 'response_assistant.txt'))
        image_up              = camera1.get_image_bundle()
        rgb_up                = image_up['rgb']
        playsound(1000, 300)
        rgb_up                = cv2.cvtColor(rgb_up, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(file_path, file_name1), rgb_up)
        cv2.imwrite(os.path.join(file_path, f'step{step}_cam1.png'), rgb_up)
        # show_image(rgb_up)
        # Getting the base64 string
        base64_image_current_step1 = encode_image(os.path.join(file_path, file_name1))
        
        image_side              = camera2.get_image_bundle()
        rgb_side                = image_side['rgb']
        playsound(1000, 300)
        rgb_side                = cv2.cvtColor(rgb_side, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(file_path, file_name2), rgb_side)
        cv2.imwrite(os.path.join(file_path, f'step{step}_cam2.png'), rgb_side)
        # show_image(rgb_side)
        # Getting the base64 string
        base64_image_current_step2 = encode_image(os.path.join(file_path, file_name2))
        
        response = client.chat.completions.create(
        model=model_name,
        messages=[

            {"role": "system","content": "You are a helpful assistant capable of recognizing the presence of an object within an image. I pass you three images: the first one contains a list of all the components and their assembly precendences that you need to locate within the second and third image. The second and third image depict the same object from different angles to help you better understand the scene. Rember that only one component has been added in compared to your previous response. In the previous step, I asked you to identify which components are in the scene and you provided the answer that I am passing to the assistant role. During the identification phase, you need to verify that all the components you have identified meet the assembly precedence requirements defined in the first image. Respond only with a list of components indicating YES if are present, NO if they are not."},
            {"role": "assistant","content": f"{text_query}"},

            {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_component_list}",
                    "detail": "high"
                },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image_current_step1}",
                    "detail": "high"
                },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image_current_step2}",
                    "detail": "high"
                },
                },
                {
                "type": "text",
                "text":"1- Chassis - Is present in the second and third image? YES or NO",
                
                },
                {
                "type": "text",
                "text":"2- Motor - Is present in the second and third image? YES or NO",
                
                },
                {
                "type": "text",
                "text":"3- Body - Is present in the second and third image? YES or NO",
                
                },
                {
                "type": "text",
                "text":"4- Wheels - Is present in the second and third image? YES or NO",
                
                },
                {
                "type": "text",
                "text":"5- Intake - Is present in the second and third image? YES or NO",
                
                },
                {
                "type": "text",
                "text":"6- Front bumper - Is present in the second and third image? YES or NO",
                
                },
                {
                "type": "text",
                "text":" 7- Roof - Is present in the second and third image? YES or NO",
                
                },
                {
                "type": "text",
                "text":"8- Spoiler - Is present in the second and third image? YES or NO",
                },
                {
                "type": "text",
                "text":"9- Lateral bumpers - Is present in the second and third image? YES or NO.",
                },
                

            ],
            },
            ],

        max_tokens=1000,
        top_p=0,
        temperature=0,
        seed=123
        )

        components_detected = response.__dict__['choices'][0].__dict__['message'].__dict__['content']
        print(components_detected)
        write_text_file(os.path.join(cwd, DATA_DIR, 'response_assistant.txt'), components_detected)
        
    
    
        
        response2 = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant capable of determining the next component for the robot to pick up and delivery to the assembly area. Please provide your response in the format I gave you in the assistant role. Only include the number obtained from the fourth reasoning step in the robot.move_to() command."},
            {"role": "assistant", "content":f"{assistant_robot_planner}"},
            
            {"role": "user", "content": f"First step: Based on the content you provided as the answer {components_detected}, resume the components that are present in a list format like [1,2,3,...] , with each component indicated with YES in your response."},

            {"role": "user", "content": f"Second step: Knowing the components already brought by the robot in {brought_objec}, make the union of them with the detected components in your first step: {brought_objec} and detected."},
        
            {"role": "user", "content": f"Third step: Subtract the set {full_set} from the set response obtained in the second step."},
            
            {"role": "user", "content": "Fourth step: : The fourth step number is the first element of the ordered set in the third step response."},
        
            {"role": "user", "content": "Fifth step: Write the output response following the format provided in the assistant role."},
        ],
        top_p=0,
        seed=123,
        max_tokens=1000
        )

        robot_movements = response2.__dict__['choices'][0].__dict__['message'].__dict__['content']
        print("Robot Movements:")
        print(robot_movements)
        msg                   = robot_movements.strip().split('\n')[0]
        msg                   = msg.split('move_to')[1][1]
        if msg=="END":
            print("END")
            break

        component = int(msg)
        assert component > 0 and component <= len(object_positions)
        object_pose =  array_cm_to_m(object_positions[component-1]) + array_deg_to_rad(tool_orientation)
        
        if component == 4:
            n = 4
        elif component == 9:
            n = 2
        else:
            n = 1

        x_offset = cm_to_m(x_axis_offset[component-1])

        for i in range(n):
            target_pose = Pose(*object_pose)
            x_offset_curr = i*x_offset

            target_pose.set_x(target_pose.x + x_offset_curr)
            target_pose.set_z(target_pose.z + cm_to_m(PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())

            target_pose.set_z(target_pose.z - cm_to_m(PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())

            UR5e.close_gripper()

            target_pose.set_z(target_pose.z + cm_to_m(PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())

            # =========
    
            target_pose = Pose(*delivery_pose_tmp)

            target_pose.set_z(target_pose.z + cm_to_m(PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())
           
            target_pose.set_z(target_pose.z - cm_to_m(PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())
            
            UR5e.open_gripper()

            target_pose.set_z(target_pose.z + cm_to_m(PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())

            # =========

            UR5e.send_joints(array_deg_to_rad(home_joints))

            if i != n-1:
                print("Press [ENTER] key to proceed...")
                input()

        # c.send(robot_movements.encode('ascii')); 
        # print("COMMANDS SENT")
        # #robot executes the commands and sends a response containing the number of component delivered
        # data        = c.recv(1024)
        # msg         = data.decode('ascii')
        # component=int(msg)
        brought_objec.append(component)
        t_fin_proc  = time.time()
        print(t_fin_proc-t_start_proc)
        if msg=="END":
            print("END")
            break
        # if msg=="END":
        #     print("END")
        #     c.close()
        #     s.close()
        #     break
        
        playsound(5000, 200)
        print("Press [ENTER] key to proceed...")
        input()
        step += 1

    t_end=time.time()
    print('T tot',t_end-t_start)



if __name__ == '__main__':
    main()