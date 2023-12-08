import cv2
import numpy as np
import sys

import cv2

def resize_image(image_path, scale_factor, output_path=None):
    """
    Resizes an image by a given scale factor using OpenCV.
    
    :param image_path: str - The path to the input image.
    :param scale_factor: float - The factor by which the image should be scaled.
    :param output_path: str - The path where the resized image will be saved. If None, the image will be displayed.
    """
    # Read the image from the file
    image = cv2.imread(image_path)
    
    # Check if image is loaded
    if image is None:
        print("Error: Could not read the image.")
        return

    # Calculate the new dimensions
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_dimensions = (new_width, new_height)

    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    # Save or display the image
    if output_path:
        cv2.imwrite(output_path, resized_image)
        print(f"The resized image is saved at {output_path}")
    else:
        cv2.imshow("Resized Image", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def template_matching(large_image_path, small_image_path, num_matches=5):
    # Load the images
    large_image = cv2.imread(large_image_path, cv2.IMREAD_UNCHANGED)
    small_image = cv2.imread(small_image_path, cv2.IMREAD_UNCHANGED)
    print(small_image.shape[2])

    cv2.imshow('Small Image', small_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    if large_image is None or small_image is None:
        print("Error: Unable to load images. Check the file paths.")
        return

    # Print image sizes
    print("Large Image Size:", large_image.shape)
    print("Small Image Size:", small_image.shape)

    if small_image.shape[2] == 4:
        # Split the image into alpha and color channels
        b, g, r, alpha = cv2.split(small_image)

        # Create a white background image
        white_background = np.ones_like(b) * 255

        # Blend the image with the white background based on the alpha channel
        b = b * alpha / 255 + white_background * (1 - alpha / 255)
        g = g * alpha / 255 + white_background * (1 - alpha / 255)
        r = r * alpha / 255 + white_background * (1 - alpha / 255)

        # Merge the channels back
        img_with_white_bg = cv2.merge([r, g, b])

        # Save or display the result
        cv2.imshow('Image with White Background', img_with_white_bg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        small_image = img_with_white_bg.copy()


    scale_factor = 0.5
    # Calculate the new dimensions
    new_width = int(small_image.shape[1] * scale_factor)
    new_height = int(small_image.shape[0] * scale_factor)
    new_dimensions = (new_width, new_height)

    # Resize the image
    small_image = cv2.resize(small_image, new_dimensions, interpolation=cv2.INTER_AREA)
    print(small_image.shape[2])



    # Convert images to grayscale
    large_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Small Image', small_image)
    cv2.imshow('Large Image Gray', large_gray)
    cv2.imshow('Small Image Gray', small_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # List of template matching methods available in OpenCV
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']

    for method in methods:
        # Apply template matching
        method_eval = eval(method)
        result = cv2.matchTemplate(large_gray, small_gray, method_eval)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if method in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
            top_left = min_loc
        else:
            top_left = max_loc

        # Diagnostic Messages
        print(f"Method: {method}")
        # Find the top 'num_matches' matches
        locations = np.where(result >= np.sort(result.ravel())[-num_matches])
        sorted_locations = sorted(zip(*locations[::-1]), key=lambda x: result[x[::-1]], reverse=True)
        display_image = large_image.copy()

        for i, loc in enumerate(sorted_locations[:num_matches]):
            top_left = loc
            bottom_right = (top_left[0] + small_image.shape[1], top_left[1] + small_image.shape[0])

            # Draw a rectangle around the found location
            cv2.rectangle(display_image, top_left, bottom_right, (0, 255 - 50*i, 50*i), 2)
            print(f"Match {i+1}: Location={top_left}, Score={result[loc[::-1]]}")

        cv2.imshow('Top Matches', display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py [path_to_large_image] [path_to_small_image]")
        return

    large_image_path = sys.argv[1]
    small_image_path = sys.argv[2]
    
    template_matching(large_image_path, small_image_path)

if __name__ == "__main__":
    main()
