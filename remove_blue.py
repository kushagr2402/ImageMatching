import cv2
import os
import argparse
import numpy as np

def process_image(image_path, output_dir):
    # Extract the filename from the input path
    filename = os.path.basename(image_path)

    # Build the full output path
    output_path = os.path.join(output_dir, 'processed_' + filename)

    # Load the image
    image = cv2.imread(image_path)

    # Check if image is loaded properly
    if image is None:
        print("Error: Image not found.")
        return

    # Create a mask for the condition
    processed_image = image

    # Light blue stripe background
    condition_mask = (image[:, :, 0] >= 180) & (image[:, :, 0] <= 240) & (image[:, :, 2] <= 70) & (image[:, :, 1] >= 50)
    processed_image[condition_mask] = [255, 255, 255]

    # Dark blue stripe background
    condition_mask = (image[:, :, 0] >= 140) & (image[:, :, 0] <= 160) & (image[:, :, 2] <= 1) & (image[:, :, 1] >= 5)
    processed_image[condition_mask] = [255, 255, 255]

    # Save the processed image
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved as {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Image processing script")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("output_dir", help="Directory to save the processed image")
    args = parser.parse_args()

    process_image(args.image_path, args.output_dir)

if __name__ == "__main__":
    main()
