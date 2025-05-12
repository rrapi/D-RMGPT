import logging

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width=1280,
                 height=720,
                 fps=30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None

    def __del__(self):
        self.pipeline.stop()
        print(f"Camera with id:{self.device_id} disconnected.")

    def check_available_devices():
        connect_device = []
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                device_info = (serial, product_line)  # (serial_number, product_line)
                connect_device.append(device_info[0])
        print(connect_device)

    def connect(self):
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.device_id)  
        print('device enabled!!')
       
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        # config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        # config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        cfg = self.pipeline.start(config)
        

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        print(f"Camera with id:{self.device_id} connected.")
        return 

    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames()

        #color_frame = frames.get_color_frame()
        #depth_frame = frames.get_depth_frame()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.first(rs.stream.color)
        color_image = np.asanyarray(color_frame.get_data())

        # aligned_depth_frame = aligned_frames.get_depth_frame()
        # depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        # depth_image *= self.scale
        # depth_image = np.expand_dims(depth_image, axis=2)

        return {
            'rgb': color_image,
            # 'depth': depth_image
        }
   

if __name__ == '__main__':
    cam = RealSenseCamera(device_id='046122251438')
    cam.connect()
    i=0
    while i<2:
        images = cam.get_image_bundle()

        rgb = images['rgb']
 
        i=i+1
