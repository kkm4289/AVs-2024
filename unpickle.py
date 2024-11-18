import pickle
import json

def make_serializable(data):
    """Convert data to a JSON-serializable format."""
    if isinstance(data, dict):
        return {make_serializable(key): make_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [make_serializable(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(make_serializable(item) for item in data)
    elif isinstance(data, set):
        return list(make_serializable(item) for item in data)
    elif isinstance(data, (int, float, str, bool)) or data is None:
        return data
    else:
        return str(data)  # Convert unknown types to string

def load_pkl_and_save_json(pkl_file_path, json_file_path):
    try:
        # Load data from the pickle file
        with open(pkl_file_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        
        # Convert data to JSON-serializable format
        serializable_data = make_serializable(data)
        
        # Save data to a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(serializable_data, json_file, indent=4)
        
        print(f"Data saved to {json_file_path}")
    
    except FileNotFoundError:
        print(f"File not found: {pkl_file_path}")
    except pickle.UnpicklingError:
        print("Error: Could not unpickle the file. The file might be corrupted or not a valid pickle file.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'your_file.pkl' with the path to your .pkl file
file_path = '../Subset/type1_subtype1_normal/ego_vehicle/calib/Town01_type001_subtype0001_scenario00003/Town01_type001_subtype0001_scenario00003_001.pkl'
load_pkl_and_save_json(file_path, '../calibration/ego_001.json')

file_path = '../Subset/type1_subtype1_normal/infrastructure/calib/Town01_type001_subtype0001_scenario00003/Town01_type001_subtype0001_scenario00003_001.pkl'
load_pkl_and_save_json(file_path, '../calibration/infrastructure_001.json')