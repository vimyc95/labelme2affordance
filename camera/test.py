import pyrealsense2 as rs
import numpy as np
import cv2

# 建立 pipeline
pipeline = rs.pipeline()
config = rs.config()

# 設定 L515 的影像串流格式
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 啟動 pipeline 並拿到 profile
profile = pipeline.start(config)

# 可選：讓裝置穩定幾幀（L515 啟動時深度不穩）
for _ in range(30):
    pipeline.wait_for_frames()

# 可選：設定自動曝光或手動曝光
# L515 的雷射強度與曝光值可用下列方式設定（需要根據環境調整）
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_sensor.set_option(rs.option.laser_power, 250)  # 雷射強度 0~1000
# depth_sensor.set_option(rs.option.exposure, 500)  # 曝光時間（μs）

try:
    while True:
        frames = pipeline.wait_for_frames()

        # 如果你需要對齊 RGB 與深度（非常推薦）
        # align = rs.align(rs.stream.color)
        # frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 將深度圖轉為可視化圖（0.03 是縮放係數，可依環境調整）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow('L515 RGB + Depth', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()