"""
Apply perspective shift to an image

Using a calibration from image1 to image2, apply perspective shift to image1 to make it look like image2.
"""

import cv2
import numpy as np
import pickle
# import ultralytics


np.set_printoptions(suppress=True)
car_cam_path = 'DeepAccident_type1_subtype2_normal/ego_vehicle/Camera_Front/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.jpg'
infra_cam_path = 'DeepAccident_type1_subtype2_normal/infrastructure/Camera_Back/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.jpg'
car_calib_path = 'DeepAccident_type1_subtype2_normal/ego_vehicle/calib/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.pkl'
infra_calib_path = 'DeepAccident_type1_subtype2_normal/infrastructure/calib/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.pkl'
infra_labels_path = 'DeepAccident_type1_subtype2_normal/infrastructure/label/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.txt'


def draw_bounding_box(image, list_corners_2d):
    """
    Draws a 3D bounding box (as projected 2D corners) on an image.

    Args:
    - image (numpy array): The image on which to draw the bounding box.
    - list_corners_2d (list of tuples):  label, list of 2D corners of the bounding box.

    Returns:
    - image (numpy array): The image with the bounding box drawn on it.
    """

    for labels, corners_2d in list_corners_2d:
        
        # Convert corners_2d to integer (pixel coordinates)
        corners_2d = [(int(corner[0]), int(corner[1])) for corner in corners_2d]

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
        cv2.putText(image, labels, corners_2d[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

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

def get_3d_bbox_corners(parts):
    """
    Given a BBox List, this function computes the 3D corners of the bounding box
    in LiDAR space based on position, dimensions, and yaw rotation.

    Args:
    - parts (list): The annotation list containing object details.

    Returns:
    - List of numpy arrays representing the 3D corners of the bounding box in LiDAR space.
    """

    # Extract position (x, y, z) and bounding box dimensions (L, W, H)
    x = float(parts[1])
    y = float(parts[2])
    z = float(parts[3])

    L = float(parts[4])  # Length
    W = float(parts[5])  # Width
    H = float(parts[6])  # Height

    # Extract yaw angle
    yaw = float(parts[7])

    # Half of the bounding box dimensions
    half_L = L / 2
    half_W = W / 2
    half_H = H / 2

    # List of 8 corners for the bounding box centered at the origin
    corners = [
        (-half_L, -half_W, -half_H),
        (half_L, -half_W, -half_H),
        (half_L, half_W, -half_H),
        (-half_L, half_W, -half_H),
        (-half_L, -half_W, half_H),
        (half_L, -half_W, half_H),
        (half_L, half_W, half_H),
        (-half_L, half_W, half_H)
    ]

    # Yaw rotation matrix for z-axis
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Rotate each corner by the yaw angle
    rotated_corners = [np.dot(rotation_matrix, corner) for corner in corners]

    # Object position in LiDAR space
    position = np.array([x, y, z])

    # Translate each rotated corner to the object's position
    translated_corners = [corner + position for corner in rotated_corners]

    return translated_corners


if __name__ == '__main__':
    car_cam = cv2.imread(car_cam_path)
    infra_cam = cv2.imread(infra_cam_path)
    infra_labels = np.loadtxt(infra_labels_path, delimiter=' ', dtype=str, skiprows=1)

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
    K_infra_back = infra_calib['intrinsic_Camera_Back']

    #ego to lidar to car_front_cam
    car_lidar_to_ego = car_calib['lidar_to_ego']
    car_ego_to_lidar = np.linalg.inv(car_lidar_to_ego)
    lidar_to_car_front = car_calib['lidar_to_Camera_Front']
    K_car_front = car_calib['intrinsic_Camera_Front']

    # get homography / transformation from infra_back_cam to car_front_cam
    transformation = lidar_to_car_front @ car_ego_to_lidar @ infra_lidar_to_ego @ infra_cam_to_lidar
    print("Transformation matrix: \n", transformation)

    P_infraback = K_infra_back @ infra_lidar_to_cam_back[:3, :]
    # print("P_infraback: \n", P_infraback)

    P_car = K_car_front @ lidar_to_car_front[:3, :]
    # print("P_car: \n", P_car)

    detections_2d = [] # for infra_back_cam
    car_detection_2d = [] # for car_front_cam

    for bbox in infra_labels:
        print("Label", bbox[0])
        corners_3d = get_3d_bbox_corners(bbox)
        # project onto infra_back_cam and show image
        corners_3d = np.array(corners_3d)
        # print("8 Corners: \n", (corners_3d))
        corners_2d = project_points(corners_3d, P_infraback)
        # print("2D Corners: \n", corners_2d)
        detections_2d.append([bbox[0], corners_2d])

        # #TODO apply transformation to 3d bounding box points from infra_back_cam to car_front_cam
        # ex = np.array([[0, 0, 0], [0,0,0]])
        # transformed_ex = np.array(
        #     [transformation @ np.hstack((point, [1])) for point in ex])
        # print("Transformed ex: \n", transformed_ex)
        # exit()

        # make homogeneous
        # then apply transformation to each point separately
        car_corners_3d = np.array(
            [transformation @ np.hstack((point, [1])) for point in corners_3d])
        # print("Transformed points: \n", car_corners_3d)

        # 3D to 2D: project onto car_front_cam
        assert np.all(car_corners_3d[:, 3] == 1)
        car_corners_3d = car_corners_3d[:, :3] # remove last column
        # print("3D Corners: \n", car_corners_3d)
        car_corners_2d = project_points(car_corners_3d, P_car)
        car_detection_2d.append([bbox[0], car_corners_2d])
                


    # BBoxes on original images
    image = draw_bounding_box(infra_cam, detections_2d)
    cv2.imwrite('infra_back_cam_bbox.jpg', image)
    print("Image saved as infra_back_cam_bbox.jpg")

    # Transformed BBoxes on car_front_cam
    image = draw_bounding_box(car_cam, car_detection_2d)
    cv2.imwrite('car_front_cam_bbox.jpg', image)
    print("Image saved as car_front_cam_bbox.jpg")