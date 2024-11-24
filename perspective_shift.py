"""
Apply perspective shift to an image

Using a calibration from image1 to image2, apply perspective shift to image1 to make it look like image2.
"""

import cv2
import numpy as np
import pickle
np.set_printoptions(suppress=True)
car_cam_path = 'DeepAccident_type1_subtype2_normal/ego_vehicle/Camera_Front/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.jpg'
infra_cam_path = 'DeepAccident_type1_subtype2_normal/infrastructure/Camera_Back/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.jpg'

car_calib_path = 'DeepAccident_type1_subtype2_normal/ego_vehicle/calib/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.pkl'
infra_calib_path = 'DeepAccident_type1_subtype2_normal/infrastructure/calib/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.pkl'

car_cam = cv2.imread(car_cam_path)
infra_cam = cv2.imread(infra_cam_path)

# resize images keeping aspect ratio
height, width, _ = car_cam.shape
new_width = 640
new_height = int(height * new_width / width)
stacked = np.vstack((cv2.resize(car_cam, (new_width, new_height)),
                    cv2.resize(infra_cam, (new_width, new_height))))
# cv2.imshow('stacked', stacked)
# cv2.waitKey(0)

# load calibration
with open(car_calib_path, 'rb') as f:
    car_calib = pickle.load(f)
with open(infra_calib_path, 'rb') as f:
    infra_calib = pickle.load(f)

# print("Car calibration: ", list(car_calib.keys()))
# print("Infra calibration: ", (infra_calib.keys()))

# infra_back_cam to lidar to ego
infra_lidar_to_cam_back = infra_calib['lidar_to_Camera_Back']
infra_cam_to_lidar = np.linalg.inv(infra_lidar_to_cam_back)  # inverse of the transformation
infra_lidar_to_ego = infra_calib['lidar_to_ego']

#ego to lidar to car_front_cam
car_lidar_to_ego = car_calib['lidar_to_ego']
car_ego_to_lidar = np.linalg.inv(car_lidar_to_ego)
lidar_to_car_front = car_calib['lidar_to_Camera_Front']

# get homography / transformation from infra_back_cam to car_front_cam
transformation = lidar_to_car_front @ car_ego_to_lidar @ infra_lidar_to_ego @ infra_cam_to_lidar
print("Transformation matrix: \n", transformation)

#TODO get bounding boxes of infra_back_cam

#TODO apply perespective shift to bounding boxes of infra_back_cam to car_front_cam

#TODO project bounding boxes of infra_back_cam to car_front_cam

#TODO draw bounding boxes on car_front_cam