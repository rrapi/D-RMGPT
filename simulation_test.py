# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:31:00 2024

@author: Franci Rrapi
"""

import constants
import os
from openai import OpenAI
import base64
from camera import RealSenseCamera
import time
import cv2
try:
    import winsound
except ImportError:
    def playsound(freq, duration):
        os.system('sudo play -nq -t alsa synth %s sin %s' % (duration/1000, freq))
        print("Beep!")
else:
    def playsound(freq, duration):
        winsound.Beep(freq, duration)
import numpy as np

# Set your OpenAI API key
import constants

os.environ["OPENAI_API_KEY"] 	  = constants.APIKEY
token 				                  = constants.GITHUB_TOKEN
endpoint 			                  = "https://models.inference.ai.azure.com"
model_name 			                = constants.MODEL_NAME

cwd 		= os.getcwd()
dataset_path	= os.path.join(cwd, 'data', 'Assembly_sequence_images')
dir_list 	= os.listdir(dataset_path)
files		= [f for f in dir_list if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith(".png")]
num_dataset	= len(files)
num_steps	= num_dataset/2

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

def showImage(image_path):
    # read the image 
    img = cv2.imread(image_path)
    # showing the image
    cv2.imshow('gfg', img)
    # waiting using waitKey method
    cv2.waitKey(0)


text_query 			              = read_text_file(os.path.join(cwd, 'data', 'response_assistant.txt'))
assistant_robot_planner 	    = read_text_file(os.path.join(cwd, 'data', 'assistant_robot_planner.txt'))

# Path to your image
image_path_components 		    = os.path.join(cwd, 'data', 'components_list.jpg')
# Getting the base64 string
base64_image_component_list 	= encode_image(image_path_components)



# client = OpenAI()	  # with OpenAI key
client = OpenAI(	    # with GitHub token
	base_url=endpoint,
	api_key=token
)

step          = 0
t_start		    = time.time()
brought_objec	= []
full_set	    = [1, 2, 3, 4, 5, 6, 7, 8, 9]
while step < num_steps:

    t_start_proc		            = time.time()
    text_query 			            = read_text_file(os.path.join(cwd, 'data', 'response_assistant.txt'))
    image1			                = os.path.join(dataset_path, f'step_{step+1}_cam1.png')
    # showImage(image1)
    print(f"Image: {image1}")
    # Getting the base64 string
    base64_image_current_step1 	= encode_image(image1)
    
    image2			                = os.path.join(dataset_path, f'step_{step+1}_cam2.png')
    # showImage(image2)
    print(f"Image: {image2}")
    # Getting the base64 string
    base64_image_current_step2 	= encode_image(image2)
    
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
              "text":"1- Upper Fusellage - Is present in the second and third image? YES or NO",
              
            },
            {
              "type": "text",
              "text":"2- Lower Fusellage - Is present in the second and third image? YES or NO",
              
            },
            {
              "type": "text",
              "text":"3- Motor - Is present in the second and third image? YES or NO",
              
            },
            {
              "type": "text",
              "text":"4- Tail wing - Is present in the second and third image? YES or NO",
              
            },
            {
              "type": "text",
              "text":"5- Propeller - Is present in the second and third image? YES or NO",
              
            },
            {
              "type": "text",
              "text":"6- Wing - Is present in the second and third image? YES or NO",
              
            },
            {
              "type": "text",
              "text":" 7- Chassis - Is present in the second and third image? YES or NO",
              
            },
            {
              "type": "text",
              "text":"8- Wheels. Is present in the second and third image? YES or NO.",
            },
            
            

          ],
        },
        ],

      max_tokens=1000,
      top_p=0,
      temperature=0,
      seed=123
    )

    components_detected=response.__dict__['choices'][0].__dict__['message'].__dict__['content']
    write_text_file(os.path.join(cwd, 'data', 'response_assistant.txt'), components_detected)
    print("Detected components:")
    print(components_detected)
    # print("RESPONSE")
    # print(response)
    
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
    robot_movements=response2.__dict__['choices'][0].__dict__['message'].__dict__['content']
    print("Robot Movements:")
    print(robot_movements)
    # print("RESPONSE2")
    # print(response2)
    print("COMMANDS SENT")
    #robot executes the commands and sends a response containing the number of component delivered
    msg                   = robot_movements.strip().split('\n')[0]
    msg                   = msg.split('move_to')[1][1]
    component_delivered   =int(msg)
    print("Delivered component n. {}".format(component_delivered))
    brought_objec.append(component_delivered)
    t_fin_proc            =time.time()
    print(t_fin_proc-t_start_proc)
    if msg=="END":
        print("END")
        break
    
    playsound(5000, 200)
    print("Press [ENTER] key to proceed...")
    input()
    step += 1
t_end=time.time()

print('T tot',t_end-t_start)
