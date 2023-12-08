import cv2
import numpy as np
import argparse

def feature_matching(large_image_path, small_image_path):
    # Load the images
    large_image = cv2.imread(large_image_path)
    small_image = cv2.imread(small_image_path)

    # Print image sizes
    print(f"Large Image Size: {large_image.shape}")
    print(f"Small Image Size: {small_image.shape}")

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(large_image, None)
    kp2, des2 = orb.detectAndCompute(small_image, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches
    result = cv2.drawMatches(large_image, kp1, small_image, kp2, matches[:10], None, flags=2)

    # Display the result
    cv2.imshow('Feature Matched Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Assuming matches is not empty and you want to print the locationc of the first match
    if matches:
        # Index of the matched keypoint in the large image
        large_img_idx = matches[0].queryIdx

        # Coordinate of the matched keypoint in the large image
        matched_pt = kp1[large_img_idx].pt

        print(f"Found location at coordinates: {matched_pt}")
    else:
        print("No matching features found.")


def main():
    parser = argparse.ArgumentParser(description='Image Feature Matching')
    parser.add_argument('large_image_path', type=str, help='Path to the large image')
    parser.add_argument('small_image_path', type=str, help='Path to the small image')
    args = parser.parse_args()

    feature_matching(args.large_image_path, args.small_image_path)

if __name__ == "__main__":
    main()
