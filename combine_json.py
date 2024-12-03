import os
import json

# Set directory containing the JSON files
PREPROCESSED_DATA_DIR = './preprocessed_data'
OUTPUT_FILE = './preprocessed_data/annotations.json'

# Combine all JSON files into one
def combine_json_files(input_dir, output_file):
    combined_data = {
        "annotations": [],
    }

    # Loop through all JSON files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    # Check if the data is a dictionary or a list
                    if isinstance(data, dict):
                        # If it's a dictionary, extract annotations
                        combined_data["annotations"].extend(data.get("annotations", []))
                    elif isinstance(data, list):
                        # If it's a list, loop through its items
                        for item in data:
                            if isinstance(item, dict):
                                combined_data["annotations"].extend(item.get("annotations", []))
                            elif isinstance(item, str):
                                # If the item is a string, treat it as a caption
                                # Wrap it inside a dictionary to maintain structure
                                combined_data["annotations"].append({"caption": item})
                            else:
                                print(f"Skipping invalid item in {file_path}: {item}")
                except json.JSONDecodeError as e:
                    print(f"Error loading {file_path}: {e}")

    # Write the combined data to a new JSON file
    with open(output_file, 'w') as output_f:
        json.dump(combined_data, output_f, indent=4)

    print(f"Combined annotations saved to {output_file}")

if __name__ == "__main__":
    combine_json_files(PREPROCESSED_DATA_DIR, OUTPUT_FILE)
