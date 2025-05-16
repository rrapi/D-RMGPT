# -*- coding: utf-8 -*-

import ailab_data.setup as setup
from utils import *
import constants
import os
import argparse
from openai import OpenAI
from pydantic import BaseModel
from camera import RealSenseCamera
import matplotlib.pyplot as plt
import time
import cv2
from typing import List
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
    


class ProductFormat(BaseModel):
   component_names: List[str]
   numbered_components: List[int]
   precedences: List[List[int]]
   sequence: List[int]
   valid_sequence: bool

class TaskFormat(BaseModel):
#    brought_components: List[int]
   next_component: int


class LLM():
    def __init__(self, model, max_tokens=1000, top_p=0, temperature=0, seed=123):
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.seed = seed

class Task():
    def __init__(self, name, domain, current_status:TaskFormat=None):
        self.name = name
        self.domain = domain
        self.current_status = current_status

    def set_task_status(self, new_status:TaskFormat):
        self.current_status = new_status

    def __str__(self):
        # res = f"[INFO] Predicted components: {self.current_status.brought_components}\n" \
        #     + f"[INFO] Next component: {self.current_status.next_component}"

        res = f"[INFO] Next component: {self.current_status.next_component}"
        
        return res 

class Product():
    def __init__(self, name, info:ProductFormat=None):
        self.name = name
        self.info = info

    def set_product_info(self, new_info:ProductFormat):
        self.info = new_info

    def get_sequence_string(self, sep=' -> '):
        return sep.join(map(str, self.info.sequence))

    def __str__(self):
        res = f"[INFO] Product name: {self.name}\n" \
            + f"[INFO] Components: {self.info.component_names}\n" \
            + f"[INFO] Dependencies: {self.info.precedences}"
        
        if self.info.valid_sequence:
            res += f"\n[INFO] Possible valid sequence: {self.get_sequence_string()}"

        return res 


class Env():
    def __init__(self, home_joints, delivery_pose, object_positions):
        self.home_joints = home_joints
        self.delivery_pose = delivery_pose
        self.object_positions = object_positions


# class Feedback():
#     def __init__(self):
#         self.OK = ""
#         self.NotOK = ""

# class Experiment():
#     def __init__(self, model, home_pose, delivery_pose, product_name, task, domain, product_info:ProductFormat=None, current_status:TaskFormat=None):
#         self.model = model
#         self.home_pose = home_pose
#         self.delivery_pose = delivery_pose
#         self.product_name = product_name
#         self.task = task
#         self.domain = domain
#         self.product_info = product_info
#         self.current_status = current_status

#     def set_model(self, model):
#         self.model = model

#     def set_product_info(self, product_info):
#         self.product_info = product_info

#     def set_current_status(self, current_status):
#         self.current_status = current_status

def main():

    parser = argparse.ArgumentParser(description='Input arguments')
    parser.add_argument('data_dir', type=str, help='Directory of experiment data')

    os.environ["OPENAI_API_KEY"]    = constants.APIKEY

    # DATA_DIR                     = 'ailab_data'
    args                            = parser.parse_args()
    DATA_DIR                        = args.data_dir

    file_name1                      = 'current_step1.png'
    file_name2                      = 'current_step2.png'
    cwd                             = os.getcwd()
    file_path                       = os.path.join(cwd, DATA_DIR, 'Assembly_sequence_images')
    base64_image_component_list 	= encode_image(os.path.join(cwd, DATA_DIR, 'components_list.jpg'))

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

    agent = LLM(model="gpt-4o")

    task = Task(name="assembly", domain="manufacturing")

    product = Product(name="toy car model")

    env = Env(home_joints=setup.home_joints, delivery_pose=setup.delivery_pose, object_positions=setup.object_positions)

    UR5e = RobotController("172.16.0.2", 0, 1)
    UR5e.open_gripper()
    UR5e.send_joints(array_deg_to_rad(env.home_joints))

    delivery_pose_tmp = array_cm_to_m(env.delivery_pose) + array_deg_to_rad(setup.tool_orientation)

    client = OpenAI()

    response = client.beta.chat.completions.parse(
        model=agent.model,
        messages=[

            # {"role": "system","content": f"You are an helpful assistant for the {task} of a {product}. In the uploaded image you can find the components of the product to be assembled and their precedence relationships. Please give as output a valid sequence of the components to be assembled in order (in the form: 1 -> 2 -> 5 and so on) as a Python list satisfying the dependencies. If a sequence cannot be found, return a False bool variable"},
            {"role": "system","content": f"You are an helpful task understanding agent for the {task.name} of a {product.name} in {task.domain} industry domain for collaborative robotics."},
            {"role": "user", "content": f"In the uploaded image you can find the components of the product and their precedence relationships. Return the names of the {product.name} components, their numbered list and the list of the precedence parts for each component (empty list in case of no dependency required). In case there exists a valid component sequence for the {task.name} satisfying the dependencies, return True and the sequence itself, otherwise False and None."},
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
                

            ],
            },
            ],
        response_format=ProductFormat,
        max_tokens=agent.max_tokens,
        top_p=agent.top_p,
        temperature=agent.temperature,
        seed=agent.seed
        )
    
    product.set_product_info(response.choices[0].message.parsed)
    # answer = response.__dict__['choices'][0].__dict__['message'].__dict__['content']
    print(product.info)


    if not product.info.valid_sequence:
       print(f"[ERR] Valid sequence not found for the {task} of the {product}.")
       return
    
    print(product)

    num_components = len(product.info.numbered_components)


    step = 1
    t_start         = time.time()
    brought_objec   = []
    full_set        = product.info.numbered_components
    num_components  = len(full_set)
    component       = None

    while True:
        if len(brought_objec) == num_components:
            if equal_set(full_set, brought_objec):
                break

        t_start_proc          = time.time()
        
        image_up              = camera1.get_image_bundle()
        playsound(1000, 300)
        image_up['rgb']       = cv2.cvtColor(image_up['rgb'], cv2.COLOR_BGR2RGB)
        image_up['rgb']       = increase_brightness(image_up['rgb'], value=30)
        cv2.imwrite(os.path.join(file_path, file_name1), image_up['rgb'])
        cv2.imwrite(os.path.join(file_path, f'step{step}_cam1.png'), image_up['rgb'])
        # show_image(image_up['rgb'])
        # Getting the base64 string
        base64_image_current_step1 = encode_image(os.path.join(file_path, file_name1))
        
        image_side              = camera2.get_image_bundle()
        playsound(1000, 300)
        image_side['rgb']       = cv2.cvtColor(image_side['rgb'], cv2.COLOR_BGR2RGB)
        image_side['rgb']       = increase_brightness(image_side['rgb'], value=30)
        cv2.imwrite(os.path.join(file_path, file_name2), image_side['rgb'])
        cv2.imwrite(os.path.join(file_path, f'step{step}_cam2.png'), image_side['rgb'])
        # show_image(image_side['rgb'])
        # Getting the base64 string
        base64_image_current_step2 = encode_image(os.path.join(file_path, file_name2))


        response = client.beta.chat.completions.parse(
            model=agent.model,
            messages=[

                # {"role": "system","content": f"You are an helpful assistant for the {task} of a {product}. In the uploaded image you can find the components of the product to be assembled and their precedence relationships. Please give as output a valid sequence of the components to be assembled in order (in the form: 1 -> 2 -> 5 and so on) as a Python list satisfying the dependencies. If a sequence cannot be found, return a False bool variable"},
                {"role": "system","content": f"You are an helpful reasoning and planning agent for the {task.name} of a {product.name} in {task.domain} industry domain for collaborative robotics."},
                # {"role": "user", "content": f"In the first uploaded image you can find the components of the finished product."},
                # {"role": "user", "content": f"The task understanding agent perceived and extracted {product.name} components information, saved in {product.info}."},
                {"role": "user", "content": f"In the first uploaded image you can find the components of the product and their precedence relationships."},
                {"role": "user", "content": f"The second and third uploaded images depict the current product status during the {task.name} process from different angles to help you better understand the scene."},
                {"role": "user", "content": f"Predict the next component that needs to be processed by the human operator for the {task.name} task. The next component must not belong to {brought_objec} and must not violate the precedence relationships of the {product.name}."},
                # {"role": "user", "content": f"Predict the {product.name} components present in the current {task.name} status and the next component that needs to be processed."},
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
                    

                ],
                },
                ],
            response_format=TaskFormat,
            max_tokens=1000,
            top_p=0,
            temperature=0,
            seed=123
            )
        task.set_task_status(response.choices[0].message.parsed)
        # answer = response.__dict__['choices'][0].__dict__['message'].__dict__['content']
        # print(answer)

        print(task)

        # predicted_components = task.current_status.brought_components
        
        # if not equal_set(predicted_components, brought_objec):
        #     print("[WARNING] Incompatibility")

        #     diff_elements = [e not in brought_objec for e in predicted_components]

        #     response = client.beta.chat.completions.parse(
        #         model=agent.model,
        #         messages=[

        #             # {"role": "system","content": f"You are an helpful assistant for the {task} of a {product}. In the uploaded image you can find the components of the product to be assembled and their precedence relationships. Please give as output a valid sequence of the components to be assembled in order (in the form: 1 -> 2 -> 5 and so on) as a Python list satisfying the dependencies. If a sequence cannot be found, return a False bool variable"},
        #             {"role": "system","content": f"You are an helpful feedback agent with the goal to correct and improve the output of the reasoning and planning agent for the {task.name} of a {product.name} in {task.domain} industry domain for collaborative robotics."},
        #             # {"role": "user", "content": f"In the first uploaded image you can find the components of the finished product."},
        #             # {"role": "user", "content": f"The task understanding agent perceived and extracted {product.name} components information, saved in {product.info}."},
        #             {"role": "user", "content": f"In the first uploaded image you can find the components of the product and their precedence relationships."},
        #             {"role": "user", "content": f"The second and third uploaded images depict the current product status during the {task.name} process from different angles to help you better understand the scene."},
        #             {"role": "user", "content": f"The incorrect answer of the reasoning and planning agent relies on the incompatibility between the predicted components in the current {task.name} status ({predicted_components}) and the brought components until that moment ({brought_objec})."},
        #             {"role": "user", "content": f"Taking into account the last incorrect prediction of the components {diff_elements} and based on the perception made in the first step by the task understanding agent, predict the {product.name} components present in the current {task.name} status and the next component to be processed."},
        #             {
        #             "role": "user",
        #             "content": [
        #                 {
        #                 "type": "image_url",
        #                 "image_url": {
        #                     "url": f"data:image/jpeg;base64,{base64_image_component_list}",
        #                     "detail": "high"
        #                 },
        #                 },
        #                 {
        #                 "type": "image_url",
        #                 "image_url": {
        #                     "url": f"data:image/png;base64,{base64_image_current_step1}",
        #                     "detail": "high"
        #                 },
        #                 },
        #                 {
        #                 "type": "image_url",
        #                 "image_url": {
        #                     "url": f"data:image/png;base64,{base64_image_current_step2}",
        #                     "detail": "high"
        #                 },
        #                 },
                        

        #             ],
        #             },
        #             ],
        #         response_format=TaskFormat,
        #         max_tokens=1000,
        #         top_p=0,
        #         temperature=0,
        #         seed=123
        #         )
        #     task.set_task_status(response.choices[0].message.parsed)

        #     print(task)
        #     print(f"Brought components: {brought_objec}")

        #     predicted_components = task.current_status.brought_components
            
        
        component = task.current_status.next_component
        assert component > 0 and component <= num_components
        assert component in full_set

        # response2 = client.beta.chat.completions.parse(
        
        #     model=model_name,
        #     messages=[

        #         # {"role": "system","content": f"You are an helpful assistant for the {task} of a {product}. In the uploaded image you can find the components of the product to be assembled and their precedence relationships. Please give as output a valid sequence of the components to be assembled in order (in the form: 1 -> 2 -> 5 and so on) as a Python list satisfying the dependencies. If a sequence cannot be found, return a False bool variable"},
        #         {"role": "system","content": f"You are an helpful planning agent assistant for the {task} of a {product} in {domain} industry domain for collaborative robotics."}, # The first image represents the components of the product to be assembled. The two uploaded images depict the current product assembly status from different angles to help you better understand the scene. Based on the component list and the precedence relationships of the {product}, which components in the current assembly status of the product you recognize?"}, 
        #         {"role": "user","content": f"The two uploaded images depict the current product {task} status in the {task} process from different angles to help you better understand the scene. Based on the components recognized from the two images and the previous answer of the task understanding agent, return the list of the {product} components already brought to the human operator and the next component number that meeds to be passed to the human operator for the {task}."},
        #         {
        #         "role": "user",
        #         "content": [
        #             # {
        #             # "type": "image_url",
        #             # "image_url": {
        #             #     "url": f"data:image/jpeg;base64,{base64_image_component_list}",
        #             #     "detail": "high"
        #             # },
        #             # },
        #             {
        #             "type": "image_url",
        #             "image_url": {
        #                 "url": f"data:image/png;base64,{base64_image_current_step1}",
        #                 "detail": "high"
        #             },
        #             },
        #             {
        #             "type": "image_url",
        #             "image_url": {
        #                 "url": f"data:image/png;base64,{base64_image_current_step2}",
        #                 "detail": "high"
        #             },
        #             },
                    

        #         ],
        #         },
        #         ],
        #     response_format=TaskFormat,
        #     max_tokens=1000,
        #     top_p=0,
        #     temperature=0,
        #     seed=123
        #     )
        
        # answer2 = response2.__dict__['choices'][0].__dict__['message'].__dict__['content']
        # print(answer2)

        object_pose =  array_cm_to_m(env.object_positions[component-1]) + array_deg_to_rad(setup.tool_orientation)
        
        if component == 4:
            n = 4
        elif component == 9:
            n = 2
        else:
            n = 1

        x_offset = cm_to_m(setup.x_axis_offset[component-1])

        for i in range(n):
            target_pose = Pose(*object_pose)
            x_offset_curr = i*x_offset

            target_pose.set_x(target_pose.x + x_offset_curr)
            target_pose.set_z(target_pose.z + cm_to_m(setup.PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())

            target_pose.set_z(target_pose.z - cm_to_m(setup.PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())

            UR5e.close_gripper()

            target_pose.set_z(target_pose.z + cm_to_m(setup.PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())

            # =========
    
            target_pose = Pose(*delivery_pose_tmp)

            target_pose.set_z(target_pose.z + cm_to_m(setup.PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())
           
            target_pose.set_z(target_pose.z - cm_to_m(setup.PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())
            
            UR5e.open_gripper()

            target_pose.set_z(target_pose.z + cm_to_m(setup.PICK_HEIGHT))
            UR5e.send_pose(target_pose.get_pose())

            # =========

            UR5e.send_joints(array_deg_to_rad(env.home_joints))

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
        print(f"[INFO] Brought components: {brought_objec}")
        t_fin_proc  = time.time()
        print('t = ',t_fin_proc-t_start_proc)
        
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