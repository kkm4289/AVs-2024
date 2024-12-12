import numpy as np
import cv2
import time
import pickle


def compute_transformation_matrix(car_calib, infra_calib) :
    return (car_calib['lidar_to_Camera_FrontLeft'] @
            np.linalg.inv(car_calib['lidar_to_ego']) @
            np.linalg.inv(car_calib['ego_to_world']) @
            infra_calib['ego_to_world'] @
            infra_calib['lidar_to_ego'])


def get_3d_bbox_corners(box_params):
    x, y, z = float(box_params[1]), float(box_params[2]), float(box_params[3])
    l, w, h = float(box_params[4]), float(box_params[5]), float(box_params[6])
    yaw = float(box_params[7])

    corners = np.array([
        [-l / 2, -w / 2, -h / 2], [l / 2, -w / 2, -h / 2], [l / 2, w / 2, -h / 2], [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, h / 2], [l / 2, -w / 2, h / 2], [l / 2, w / 2, h / 2], [-l / 2, w / 2, h / 2]
    ])

    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = corners @ rotation.T
    corners += np.array([x, y, z])
    return corners


def transform_points(points, transformation):
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed = points_h @ transformation.T
    return transformed[:, :3] / transformed[:, 3:]


def calculate_3d_iou(box1, box2):
    min1, max1 = np.min(box1, axis=0), np.max(box1, axis=0)
    min2, max2 = np.min(box2, axis=0), np.max(box2, axis=0)

    intersect_min = np.maximum(min1, min2)
    intersect_max = np.minimum(max1, max2)

    if np.any(intersect_max < intersect_min):
        return 0.0

    intersection = np.prod(intersect_max - intersect_min)
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)
    union = volume1 + volume2 - intersection
    return intersection / union if union > 0 else 0


def validate_alignment(infra_boxes, car_boxes, transformation) :
    start_time = time.perf_counter()
    transformed_boxes = []

    for box in infra_boxes:
        corners = get_3d_bbox_corners(box)
        transformed = transform_points(corners, transformation)
        transformed_boxes.append(transformed)

    iou_scores = []
    matches = 0
    for t_box in transformed_boxes:
        for c_box in car_boxes:
            c_corners = get_3d_bbox_corners(c_box)
            iou = calculate_3d_iou(t_box, c_corners)
            iou_scores.append(iou)
            if iou > 0.5:
                matches += 1

    end_time = time.perf_counter()

    return {
        "alignment_accuracy": matches / len(infra_boxes) if len(infra_boxes) > 0 else 0,
        "average_iou": np.mean(iou_scores) if iou_scores else 0,
        "processing_time_ms": (end_time - start_time) * 1000
    }


def measure_detection_improvement(car_detections, combined_detections) :
    car_count = len(car_detections)
    combined_count = len(combined_detections)
    improvement = ((combined_count - car_count) / car_count * 100) if car_count > 0 else 0
    return {
        "car_only_detections": car_count,
        "combined_detections": combined_count,
        "improvement_percentage": improvement
    }


def run_validation(base_dir: str, frame: str) :
    town = "Town04_type001_subtype0002_scenario00017"
    infra_label = f"{town}_{frame}.txt"
    infra_calib = f"{town}_{frame}.pkl"
    other_label = f"otherve/{town}_{frame}.txt"
    other_calib = f"otherve/{town}_{frame}.pkl"

    # Load calibration data
    with open(infra_calib, 'rb') as f:
        infra_calib_data = pickle.load(f)
    with open(other_calib, 'rb') as f:
        car_calib_data = pickle.load(f)

    transformation = compute_transformation_matrix(car_calib_data, infra_calib_data)

    # Load labels
    infra_boxes = np.loadtxt(infra_label, delimiter=' ', dtype=str, skiprows=1)
    other_boxes = np.loadtxt(other_label, delimiter=' ', dtype=str, skiprows=1)

    alignment_results = validate_alignment(infra_boxes, other_boxes, transformation)
    detection_results = measure_detection_improvement(
        other_boxes, np.concatenate([other_boxes, infra_boxes])
    )

    return {
        "alignment_metrics": alignment_results,
        "detection_metrics": detection_results
    }


if __name__ == "__main__":
    results = run_validation(".", "001")  #update frame
    print("\nValidation Results:")
    print("-" * 50)
    for metric_type, metrics in results.items():
        print(f"\n{metric_type}:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
