import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import cv2

"""
Dictionary keys for the matrices inside the .pkl files:

ego_to_world
intrinsic_Camera_Back
intrinsic_Camera_BackLeft
intrinsic_Camera_BackRight
intrinsic_Camera_Front
intrinsic_Camera_FrontLeft
intrinsic_Camera_FrontRight
lidar_to_Camera_Back
lidar_to_Camera_BackLeft
lidar_to_Camera_BackRight
lidar_to_Camera_Front
lidar_to_Camera_FrontLeft
lidar_to_Camera_FrontRight
lidar_to_ego
"""

# pkl_path_start = "../DeepAccident_mini_subset/type1_subtype1_normal/ego_vehicle/calib/Town10HD_type001_subtype0001_scenario00014/"
# sample_pkl = "Town10HD_type001_subtype0001_scenario00014_001.pkl"
# sample_calib_path = pkl_path_start + sample_pkl

# image_path_start = "../DeepAccident_mini_subset/type1_subtype1_normal/ego_vehicle/Camera_Back/Town10HD_type001_subtype0001_scenario00014/"
# sample_img = "Town10HD_type001_subtype0001_scenario00014_001.jpg"
# sample_img_path = image_path_start + sample_img

output_path = "sample_001_output.jpg"

infra_cam_path = 'DeepAccident_type1_subtype2_normal/infrastructure/Camera_Back/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.jpg'
infra_calib_path = 'DeepAccident_type1_subtype2_normal/infrastructure/calib/Town04_type001_subtype0002_scenario00017/Town04_type001_subtype0002_scenario00017_002.pkl'
sample_calib_path = infra_calib_path
sample_img_path = infra_cam_path


def load_pkl_file(file_path):
    """
    Loads and inspects the contents of a .pkl file.

    :param file_path: Path to the .pkl file.
    :return: None (prints the contents to the console).
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        return data

def get_3d_bbox_corners(annotation):
    """
    Given an annotation string, this function computes the 3D corners of the bounding box
    in LiDAR space based on position, dimensions, and yaw rotation.

    Args:
    - annotation (str): The annotation string containing object details.

    Returns:
    - List of numpy arrays representing the 3D corners of the bounding box in LiDAR space.
    """
    # Parse the annotation string into components
    parts = annotation.split()
    print(parts)

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

def draw_bounding_box(image, corners_2d):
    """
    Draws a 3D bounding box (as projected 2D corners) on an image.

    Args:
    - image (numpy array): The image on which to draw the bounding box.
    - corners_2d (list of tuples): 2D points representing the corners of the bounding box.

    Returns:
    - image (numpy array): The image with the bounding box drawn on it.
    """
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

    return image


# sample annotation, get these from the files
# annotation_sample = ("car -13.459459497478285 -0.0037344838527602064 "
#                      "-1.2119891287962656 4.791779518127441 2.163450002670288 "
#                      "1.4876600503921509 0.00033209590742323136 -4.587319957896956 "
#                      "-0.016017722274652455 5481 197 True")


annotation_sample = ("car -59.24394974113744 74.86522934894205 -2.6113033294677734 4.604510307312012 1.9315935373306274 1.854846715927124 -0.003082606357998318 0.0 0.0 20014 0 False")

# Example usage
matrices_01 = load_pkl_file(sample_calib_path)

# Example matrices (replace these with actual matrices from the .pkl file)
# K = np.array([[800, 560.166, 0],
#               [450, 0, -560.166],
#               [1, 0, 0]])  # Intrinsic matrix for a specific camera

K = matrices_01["intrinsic_Camera_Back"]

# T = np.array([[-9.99999953e-01, -8.76397347e-08,  1.28084908e-11, -2.09641438e+00],
#               [8.74490193e-08, -1.00000001e+00,  3.90177943e-10, -1.58926697e-06],
#               [-1.28007402e-10, 3.90494618e-10,  9.99999942e-01,  2.99999657e-01],
#               [0, 0, 0, 1]])  # Transformation from LiDAR to the same camera

T = matrices_01["lidar_to_Camera_Back"]

# Extract the 3x4 portion of T (rotation + translation)
T_camera = T[:3, :]

# Calculate the projection matrix P
P = K @ T_camera

print("Projection matrix (P):")
print(P)


bbox_corners = get_3d_bbox_corners(annotation_sample)

points_2d = project_points(np.array(bbox_corners), P)

print("2D Pixel coordinates:")
print(points_2d)

image = cv2.imread(sample_img_path)
projected_corners = points_2d

image_with_bbox = draw_bounding_box(image, projected_corners)

cv2.imshow("Image with Bounding Box", image_with_bbox)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(output_path, image_with_bbox)




