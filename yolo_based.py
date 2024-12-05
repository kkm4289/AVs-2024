"""
Apply perspective shift to an image

Using a calibration from image1 to image2, apply perspective shift to image1 to make it look like image2.
"""

import cv2
import numpy as np
import pickle
# import ultralytics
import torch
import cv2
import numpy as np


np.set_printoptions(suppress=True)
car_cam_path = 'DeepAccident_type1_subtype2_normal/ego_vehicle/Camera_Front/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.jpg'
infra_cam_path = 'DeepAccident_type1_subtype2_normal/infrastructure/Camera_Back/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.jpg'
car_calib_path = 'DeepAccident_type1_subtype2_normal/ego_vehicle/calib/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.pkl'
infra_calib_path = 'DeepAccident_type1_subtype2_normal/infrastructure/calib/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.pkl'
infra_labels_path = 'DeepAccident_type1_subtype2_normal/infrastructure/label/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.txt'


def draw_bounding_box(image, list_corners_2d):
    """
    Draws a 3D bounding box (as projected 2D corners) on an image.
    """

    for labels, corners_2d in list_corners_2d:

        # Convert corners_2d to integer (pixel coordinates)
        corners_2d = [(int(corner[0]), int(corner[1]))
                      for corner in corners_2d]

        # Draw lines between the corners to form the bounding box
        for i in range(4):
            # Draw lines for the bottom and top faces of the bounding box
            cv2.line(image, corners_2d[i], corners_2d[(i + 1) % 4], (0, 255, 0),
                     2)  # Bottom
            cv2.line(image, corners_2d[i + 4], corners_2d[(i + 1) % 4 + 4],
                     (0, 255, 0), 2)  # Top

        # Draw the vertical lines connecting top and bottom faces
        for i in range(4):
            cv2.line(image, corners_2d[i], corners_2d[i + 4], (0, 255, 0), 2)

        # Draw label
        cv2.putText(
            image, labels, corners_2d[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return image

def draw_bounding_boxes(image, boxes, class_names):
    height, width, _ = image.shape

    # Draw bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        class_id = int(box[5])
        confidence = box[4]

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Put the class name and confidence on the bounding box
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

def project_points(points_3d, P):
    """
    Projects 3D points onto the 2D image plane using a projection matrix.

    :param points_3d: Nx3 numpy array of 3D points in LiDAR space.
    :param P: 3x4 projection matrix.
    :return: Nx2 numpy array of 2D pixel coordinates.
    """
    # Add a column of ones to the points to make them homogeneous
    points_homogeneous = np.hstack(
        (points_3d, np.ones((points_3d.shape[0], 1))))

    # Project onto 2D image plane
    points_2d_homogeneous = points_homogeneous @ P.T

    # Normalize to get pixel coordinates
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2][:,
                                                                           np.newaxis]
    return points_2d


if __name__ == '__main__':
    car_cam = cv2.imread(car_cam_path)
    infra_cam = cv2.imread(infra_cam_path)
    # infra_labels = np.loadtxt(infra_labels_path, delimiter=' ', dtype=str, skiprows=1)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    results = model(infra_cam_path)
    yolo_boxes = results.xyxy[0].numpy()  # xyxy format


    # resize images keeping aspect ratio
    height, width, _ = car_cam.shape
    new_width = 640
    new_height = int(height * new_width / width)
    stacked = np.vstack((cv2.resize(car_cam, (new_width, new_height)),
                        cv2.resize(infra_cam, (new_width, new_height))))

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
    K_infra_back = infra_calib['intrinsic_Camera_Back']

    #ego to lidar to car_front_cam
    car_lidar_to_ego = car_calib['lidar_to_ego']
    car_ego_to_lidar = np.linalg.inv(car_lidar_to_ego)
    lidar_to_car_front = car_calib['lidar_to_Camera_Front']
    K_car_front = car_calib['intrinsic_Camera_Front']

    # get homography / transformation from infra_back_cam to car_front_cam
    # infra cam to lidar to ego to car lidar to car cam
    transformation = lidar_to_car_front @ car_ego_to_lidar @ infra_lidar_to_ego
    # transformation = np.identity(4)


    # print("Transformation matrix: \n", transformation)

    P_infraback = K_infra_back @ infra_lidar_to_cam_back[:3, :]
    # print("P_infraback: \n", P_infraback)

    P_car = K_car_front @ np.identity(4)[:3, :]

    # print("P_car: \n", P_car)

    detections_2d = [] # for infra_back_cam
    car_detection_2d = [] # for car_front_cam
    bbox = yolo_boxes[2]

    im = draw_bounding_boxes(infra_cam, [bbox], model.names)
    cv2.imwrite('bbox.jpg', im)
    # Corners 2d should be 3 x 4 in format 
    corners_2d = np.array([[bbox[0], bbox[1], 1], [bbox[2], bbox[1], 1], [bbox[2], bbox[3], 1], [bbox[0], bbox[3], 1]]).T
    print("corners_2d: \n", corners_2d)

    #TODO  unproject 2D bounding box to 3D
    corners_3d_infra = np.linalg.inv(
        K_infra_back) @ corners_2d  # Shape: (3, 4)
    corners_3d_infra = np.vstack((corners_3d_infra, np.ones(
        (1, corners_3d_infra.shape[1]))))  # Add homogenous coord
    print("corners_3d_infra: \n", corners_3d_infra)

    #TODO display 3D bounding box on the image

    #TODO Transform 3D bounding box to car_front_cam

    #TODO display 3D bounding box on the car_front_cam image

    #TODO Project 3D points to 2D car_front_cam
    


    # BBoxes on original images
    infra_bboxes = draw_bounding_boxes(infra_cam, yolo_boxes, model.names)
    cv2.imwrite('sample/infra_back_cam_bbox.jpg', infra_bboxes)
    print("Image saved as infra_back_cam_bbox.jpg")

    # # Transformed BBoxes on car_front_cam
    # image = draw_bounding_box(car_cam, car_detection_2d)
    # cv2.imwrite('sample/car_front_cam_bbox.jpg', image)
    # print("Image saved as car_front_cam_bbox.jpg")