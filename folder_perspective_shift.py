"""
Apply perspective shift to an image

Using a calibration from image1 to image2, apply perspective shift to image1 to make it look like image2.
"""

#TODO make run for all frames in a scene
#TODO benchmark time

import cv2
import numpy as np
import pickle
import os
import torch
import time
import statistics

np.set_printoptions(suppress=True)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

# ------------------------------------------------------------------------------

# TODO: choose scene here
scene = 2

if scene == 1:
    print("Scene " + str(scene) + ", HD version")
    # First subtype (HD)
    # ego vehicle camera front right
    # infrastructure camera back
    subtype_num = "1"
    car = "ego_vehicle"  # can be ego_vehicle, ego_vehicle_behind, other_vehicle
    car_cam_direction = "FrontLeft"
    infra_cam_direction = "Front"
    town = "Town10HD_type001_subtype0001_scenario00014"  # images use this, but with _001 at the end
    frame_num = "005"  # using a single frame for testing purposes
else:
    print("Scene " + str(scene) + ", non-HD version")
    # Second subtype (non-HD version)
    # other vehicle camera front left
    # infrastructure camera front
    # frame 005
    subtype_num = "2"
    car = "other_vehicle"  # can be infrastructure, ego_vehicle, ego_vehicle_behind, etc.
    car_cam_direction = "FrontLeft"
    infra_cam_direction = "Front"
    town = "Town04_type001_subtype0002_scenario00017"  # images use this, but with _001 at the end
    frame_num = "099"  # using a single frame for testing purposes

# ------------------------------------------------------------------------------
    # Default paths
    infra = "infrastructure"
    image_output_path = "output_subtype_" + subtype_num + "/"  # this is the folder where the output images go

    cam_folder_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + car + '/Camera_' + car_cam_direction + '/' + town
    car_cam_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + car + '/Camera_' + car_cam_direction + '/' + town + '/' + town + '_' + frame_num + '.jpg'
    car_calib_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + car + '/calib/' + town + '/' + town + '_' + frame_num + '.pkl'

    infra_cam_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + infra + '/Camera_' + infra_cam_direction + '/' + town + '/' + town + '_' + frame_num + '.jpg'
    infra_calib_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + infra + '/calib/' + town + '/' + town + '_' + frame_num + '.pkl'
    infra_labels_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + infra + '/label/' + town + '/' + town + '_' + frame_num + '.txt'

# ------------------------------------------------------------------------------



def init_paths(frame_num):
    global cam_folder_path, car_cam_path, car_calib_path, infra_cam_path, infra_calib_path, infra_labels_path

    infra = "infrastructure"
    image_output_path = "output_subtype_" + subtype_num + "/"  # this is the folder where the output images go

    cam_folder_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + car + '/Camera_' + car_cam_direction + '/' + town
    car_cam_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + car + '/Camera_' + car_cam_direction + '/' + town + '/' + town + '_' + frame_num + '.jpg'
    car_calib_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + car + '/calib/' + town + '/' + town + '_' + frame_num + '.pkl'

    infra_cam_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + infra + '/Camera_' + infra_cam_direction + '/' + town + '/' + town + '_' + frame_num + '.jpg'
    infra_calib_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + infra + '/calib/' + town + '/' + town + '_' + frame_num + '.pkl'
    infra_labels_path = 'DeepAccident_type1_subtype' + subtype_num + '_normal/' + infra + '/label/' + town + '/' + town + '_' + frame_num + '.txt'


def draw_bounding_box(image, list_corners_2d, color=(0, 255, 0)):
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
            cv2.line(image, corners_2d[i], corners_2d[(i + 1) % 4], color,
                    2)  # Bottom
            cv2.line(image, corners_2d[i + 4], corners_2d[(i + 1) % 4 + 4],
                    color, 2)  # Top

        # Draw the vertical lines connecting top and bottom faces
        for i in range(4):
            cv2.line(image, corners_2d[i], corners_2d[i + 4], color, 2)

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

def yolo_check (bbox, image, yolo_boxes):
    """

    """

    # convert to yolo format by taking the max and min of the corners
    x_min = np.min(bbox[:, 0])
    y_min = np.min(bbox[:, 1])
    x_max = np.max(bbox[:, 0])
    y_max = np.max(bbox[:, 1])
    bbox = [x_min, y_min, x_max, y_max]

    for yolo_box in yolo_boxes:
        # if bbox is near any yolo box, return True
        if abs(yolo_box[0] - bbox[0]) < 8 or abs(yolo_box[1] - bbox[1]) < 8:
            return True
    return False

def perspective_shift():
    print("running...")

    car_cam = cv2.imread(car_cam_path)
    infra_cam = cv2.imread(infra_cam_path)
    infra_labels = np.loadtxt(infra_labels_path, delimiter=' ', dtype=str, skiprows=1)

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

    model = torch.hub.load('ultralytics/yolov5',
                               'yolov5s', pretrained=True)
    results = model(infra_cam_path)
    yolo_boxes = results.xyxy[0].numpy()  # xyxy format

#     # infra_back_cam to lidar to ego
    infra_lidar_to_cam = infra_calib['lidar_to_Camera_Front']
    K_infra = infra_calib['intrinsic_Camera_Front']
    K_car = car_calib['intrinsic_Camera_FrontLeft']

    # --------------------------------------------------------------------------
    # ChatGPT code

    # Step 1: Infrastructure LiDAR to Infrastructure Ego Frame
    infra_lidar_to_ego = infra_calib[
        'lidar_to_ego']  # Transform LiDAR -> Infra Ego

    # Step 2: Infrastructure Ego to World Frame
    infra_ego_to_world = infra_calib[
        'ego_to_world']  # Transform Infra Ego -> World

    # Step 3: World to Car Ego Frame
    car_world_to_ego = np.linalg.inv(
        car_calib['ego_to_world'])  # Transform World -> Car Ego

    # Step 4: Car Ego to Car LiDAR Frame
    car_lidar_to_ego = car_calib[
        'lidar_to_ego']  # Transform Car LiDAR -> Car Ego
    car_ego_to_lidar = np.linalg.inv(
        car_lidar_to_ego)  # Transform Car Ego -> Car LiDAR

    # Step 5: Car LiDAR to Front-Left Camera
    car_lidar_to_camera = car_calib[
        'lidar_to_Camera_FrontLeft']  # Transform Car LiDAR -> Front-Left Camera

    # Final Transformation: Infra LiDAR -> Front-Left Camera
    # T_infra_to_car_camera = (
    transformation = (
            car_lidar_to_camera @  # Step 5: Car LiDAR -> Camera
            car_ego_to_lidar @  # Step 4: Car Ego -> LiDAR
            car_world_to_ego @  # Step 3: World -> Car Ego
            infra_ego_to_world @  # Step 2: Infra Ego -> World
            infra_lidar_to_ego  # Step 1: Infra LiDAR -> Infra Ego
    )

    # --------------------------------------------------------------------------

    P_infraback = K_infra @ infra_lidar_to_cam[:3, :]
    # print("P_infraback: \n", P_infraback)


    P_car = K_car[:3, :] @ np.identity(4)[:3, :]
    # print("P_car: \n", P_car)

    detections_2d = [] # for infra_back_cam
    car_detection_2d = [] # for car_front_cam

    for bbox in infra_labels:
        print("Label", bbox[0])
        corners_3d = get_3d_bbox_corners(bbox)
        # project onto infra_back_cam and show image
        corners_3d = np.array(corners_3d)

        # print("3D Corners: \n", (corners_3d))
        corners_2d = project_points(corners_3d, P_infraback)
        if not yolo_check(corners_2d, infra_cam, yolo_boxes):
            print("BBox not in YOLO. Might not be visable")
            continue

        print("2D Corners: \n", corners_2d)
        detections_2d.append([bbox[0], corners_2d])

        # make homogeneous then apply transformation to each point separately
        car_corners_3d = []
        for point in corners_3d:
            car_corners_3d.append(transformation @ np.hstack((point, [1])))
        car_corners_3d = np.array(car_corners_3d)
        print("Transformed Car Corners 3D: \n", car_corners_3d)

        # 3D to 2D: project onto car_front_cam
        car_corners_3d = car_corners_3d[:, :3] # remove last column
        # print("3D Corners: \n", car_corners_3d)
        car_corners_2d = project_points(car_corners_3d, P_car)
        car_detection_2d.append([bbox[0], car_corners_2d])


    # BBoxes on original images
    image = draw_bounding_box(infra_cam, detections_2d, green)
    # cv2.imwrite('sample3/infra_cam_bbox.jpg', image)
    image_type = "infra_cam_bbox"
    image_file_name = image_output_path + image_type + "_" + frame_num + ".jpg"
    cv2.imwrite(image_file_name, image)
    print("Image saved as", image_file_name)

    # Transformed BBoxes on car_front_cam
    image = draw_bounding_box(car_cam, car_detection_2d, green)
    # TODO compute car's lidar BBoxes in red and perspected shifted BBoxes in green



    image_type = "car_cam_bbox"
    image_file_name = image_output_path + image_type + "_" + frame_num + ".jpg"
    # cv2.imwrite('sample3/car_cam_bbox.jpg', image)
    cv2.imwrite(image_file_name, image)
    print("Image saved as", image_file_name)

if __name__ == "__main__":

    # get list of frames in camera folder
    frames = os.listdir(cam_folder_path)
    frames = [(frame.split('_')[-1].split('.')[0]) for frame in frames]
    frames = sorted(frames)
    # print("Frames in the folder:", frames)

    # frames = frames[:5]

    times = []
    for frame in frames:
        frame_num = str(frame).zfill(3)
        start_time = time.time()
        init_paths(frame_num)
        perspective_shift()

        end_time = time.time()
        time_taken = end_time - start_time
        times.append(time_taken)
        # print("Shifted perspective for frame", frame_num)


    total_time = sum(times)
    average_time = total_time / len(times)
    std_dev_time = statistics.stdev(times)
    filename = town + "_timing_results.txt"

    with open("timing_results.txt", "w") as f:
        f.write(f"Scene: {town}\n")
        f.write(f"Total time: {total_time:.2f} seconds\n")
        f.write(f"Average time: {average_time:.2f} seconds\n")
        f.write(f"Standard deviation: {std_dev_time:.2f} seconds\n")