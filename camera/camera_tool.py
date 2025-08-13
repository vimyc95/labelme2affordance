import pyrealsense2 as rs
import numpy as np
class rs_camema:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color) 

        # 取得一幀以保存內參
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        self.color_intr = color_frame.profile.as_video_stream_profile().intrinsics

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # print(depth_scale)

        return color_image, depth_image/1e3
        
    def get_K(self):
        if not hasattr(self, 'color_intr'):
            raise RuntimeError("Color intrinsics not initialized.")

        # intr = self.color_intr
        # K = np.array([
        #     [intr.fx, 0,       intr.ppx],
        #     [0,       intr.fy, intr.ppy],
        #     [0,       0,       1.0]
        # ],dtype=np.float64)
        # return '\n'.join(' '.join(f'{v:.18e}' for v in row) for row in K)
        intr = self.color_intr
        return np.array([
            [intr.fx, 0,       intr.ppx],
            [0,       intr.fy, intr.ppy],
            [0,       0,       1.0]
        ], dtype=np.float64)

if __name__ == "__main__":
    import cv2

    cap = rs_camema()

    while(1):
        rgb, depth = cap.read()
        cv2.imshow('rgb', rgb)
        cv2.imshow("depth", depth)
        cv2.waitKey(1)

    
