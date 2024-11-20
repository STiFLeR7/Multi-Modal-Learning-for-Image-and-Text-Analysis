from coco_loader import get_coco_dataloader

def main():
    # Path to your dataset
    dataset_dir = "D:/COCO-DATASET/coco2017"

    # Get the DataLoader
    dataloader = get_coco_dataloader(dataset_dir, batch_size=16, num_workers=4)

    # Iterate through the data
    for images, captions in dataloader:
        print("Batch of images shape:", images.shape)
        print("Batch of captions:", captions)
        break

if __name__ == "__main__":
    main()
