import json
import os

def fix_annotations_file(input_path, output_path):
    try:
        # Load the annotations JSON file
        with open(input_path, "r") as f:
            data = json.load(f)

        # Verify top-level keys
        required_keys = ["images", "annotations", "categories"]
        for key in required_keys:
            if key not in data:
                print(f"Missing top-level key: {key}")
                data[key] = []

        # Fix image entries
        for image in data.get("images", []):
            if "id" not in image:
                print(f"Missing 'id' in image: {image}")
                continue
            if "file_name" not in image:
                image["file_name"] = f"image_{image['id']}.jpg"
            if "height" not in image:
                image["height"] = 224  # Default height
            if "width" not in image:
                image["width"] = 224  # Default width

        # Fix annotation entries
        for annotation in data.get("annotations", []):
            if "id" not in annotation:
                print(f"Missing 'id' in annotation: {annotation}")
                continue
            if "image_id" not in annotation:
                print(f"Missing 'image_id' in annotation {annotation['id']}. Skipping.")
                continue
            if "category_id" not in annotation:
                annotation["category_id"] = 0  # Default to category 0

        # Fix categories
        if not data.get("categories"):
            print("Adding default categories.")
            data["categories"] = [{"id": i, "name": f"category_{i}"} for i in range(5)]

        # Save the fixed JSON
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Fixed JSON file saved at {output_path}")
    except Exception as e:
        print(f"Error while processing the annotations file: {e}")

if __name__ == "__main__":
    input_path = "./preprocessed_data/annotations.json"  # Input file path
    output_path = "./preprocessed_data/fixed_annotations.json"  # Output file path
    fix_annotations_file(input_path, output_path)
