import json
import os

# Function to search for a string in a JSON object and return matching values
def search_json_and_return(json_obj, target_string, key_name=None):
    matching_values = []
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if isinstance(value, (dict, list)):
                matching_values.extend(search_json_and_return(value, target_string, key_name=key))
            elif isinstance(value, str) and target_string in value:
                matching_values.append({key: value})
    elif isinstance(json_obj, list):
        for item in json_obj:
            if isinstance(item, (dict, list)):
                matching_values.extend(search_json_and_return(item, target_string, key_name=key_name))
            elif isinstance(item, str) and target_string in item:
                matching_values.append([key_name,item])
    if matching_values.__len__() > 2:
        print(f"Warning: {matching_values.__len__()} matching datasets found: {matching_values}.")
        return None
    return matching_values
def load_dataset_path(dataset_name) -> list:
    try:
        with open("datasets_files_name.json", "r") as json_file:
            json_obj = json.load(json_file)
            matching_values = search_json_and_return(json_obj, dataset_name)
            if matching_values is not None and not matching_values.__len__() == 0:
                print(f"Loading dataset: {matching_values[0]}")

                return matching_values[0]
            elif matching_values is None:
                print(f"Specify the dataset name more precisely.")
            else:
                print(f"Dataset not found.")
    except FileNotFoundError:
        print(f"Json file not found in the root directory. Please download the file using the instructions in the README.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON. File might be corrupted.")


if __name__ == '__main__':
    """Test if the datasets have been downloaded correctly
    """
    datasets = load_dataset_path("cover")
    print(datasets)
    if os.path.isfile("datasets/" + datasets[0] + "/" + datasets[1]):
        print("Datasets have been dowloaded correctly")
    else:
        print("Error downloading datasets")
