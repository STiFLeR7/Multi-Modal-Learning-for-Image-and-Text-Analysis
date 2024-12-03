import json
import os

PREPROCESSED_DATA_DIR = './preprocessed_data'  # Path to your preprocessed data
OUTPUT_FILE = './preprocessed_data/annotations.json'

def combine_json_files(preprocessed_data_dir, output_file):
    combined_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for filename in os.listdir(preprocessed_data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(preprocessed_data_dir, filename)
            with open(file_path, 'r') as f:
                try:
                    # Load the data from the file
                    data = json.load(f)
                    print(f"Processing file: {filename}")

                    # If data is a string, try to load it again as JSON
                    if isinstance(data, str):
                        print(f"Found string in {filename}, trying to load it as JSON.")
                        try:
                            data = json.loads(data)  # Try parsing the string as JSON
                        except json.JSONDecodeError:
                            print(f"Skipping invalid string in {filename}: Could not decode as JSON.")
                            continue

                    # If data is a list, iterate through it
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                annotations = item.get('annotations', [])
                                if isinstance(annotations, list):
                                    combined_data["annotations"].extend(annotations)
                                else:
                                    print(f"Skipping invalid 'annotations' in {filename}: Expected list, found {type(annotations)}")

                                images = item.get('images', [])
                                if isinstance(images, list):
                                    combined_data["images"].extend(images)
                                else:
                                    print(f"Skipping invalid 'images' in {filename}: Expected list, found {type(images)}")

                                categories = item.get('categories', [])
                                if isinstance(categories, list):
                                    combined_data["categories"].extend(categories)
                                else:
                                    print(f"Skipping invalid 'categories' in {filename}: Expected list, found {type(categories)}")
                            else:
                                print(f"Skipping invalid item in {filename}: Expected dictionary, found {type(item)}")

                    # If data is a dictionary, process normally
                    elif isinstance(data, dict):
                        annotations = data.get('annotations', [])
                        if isinstance(annotations, list):
                            combined_data["annotations"].extend(annotations)
                        else:
                            print(f"Skipping invalid 'annotations' in {filename}: Expected list, found {type(annotations)}")

                        images = data.get('images', [])
                        if isinstance(images, list):
                            combined_data["images"].extend(images)
                        else:
                            print(f"Skipping invalid 'images' in {filename}: Expected list, found {type(images)}")

                        categories = data.get('categories', [])
                        if isinstance(categories, list):
                            combined_data["categories"].extend(categories)
                        else:
                            print(f"Skipping invalid 'categories' in {filename}: Expected list, found {type(categories)}")
                    else:
                        print(f"Skipping {filename}: Expected dictionary or list structure, but got {type(data)}")

                except Exception as e:
                    print(f"Skipping invalid item in {filename}: {str(e)}")
    
    # Save the combined data
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)
    print(f"Combined annotations saved to {output_file}")

# Execute combining JSON files
combine_json_files(PREPROCESSED_DATA_DIR, OUTPUT_FILE)
