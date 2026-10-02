import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_psnr(reference_image_path, denoised_image_path):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
    reference_image_path (str): Path to the reference image.
    denoised_image_path (str): Path to the denoised (distorted) image.
    
    Returns:
    float: PSNR value.
    """
    # Load the reference and denoised images
    reference_image = cv2.imread(reference_image_path)
    denoised_image = cv2.imread(denoised_image_path)

    # Convert images to grayscale
    reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    denoised_image_gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    # Compute Mean Squared Error (MSE)
    mse = np.mean((reference_image_gray - denoised_image_gray) ** 2)

    if mse == 0:
        # PSNR is infinity if MSE is zero
        psnr = 100
    else:
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr

def calculate_psnr_for_all_images(reference_folder, denoised_folder):
    """
    Calculate PSNR for all images in the reference and denoised folders.
    
    Args:
    reference_folder (str): Path to the folder containing reference images.
    denoised_folder (str): Path to the folder containing denoised images.
    
    Returns:
    list: List of PSNR values for each pair of images.
    """
    # Get list of all images in both folders
    reference_images = os.listdir(reference_folder)
    denoised_images = os.listdir(denoised_folder)

    # Ensure the number of images in both folders is the same
    assert len(reference_images) == len(denoised_images), "Number of images in reference_folder and denoised_folder must be the same."

    psnr_values = []
    
    for ref_img_name, denoised_img_name in zip(reference_images, denoised_images):
        # Construct full image paths
        ref_img_path = os.path.join(reference_folder, ref_img_name)
        denoised_img_path = os.path.join(denoised_folder, denoised_img_name)

        # Calculate PSNR for the current pair of images
        psnr_value = calculate_psnr(ref_img_path, denoised_img_path)
        psnr_values.append(psnr_value)

        # Print PSNR value
        print(f'PSNR for {ref_img_name} and {denoised_img_name}: {psnr_value:.2f} dB')

    return psnr_values

# Example usage:
reference_folder = 'Low-images'  # Path to reference images
denoised_folder = 'High-images'  # Path to denoised images

# Calculate PSNR for all image pairs in the folders
psnr_values = calculate_psnr_for_all_images(reference_folder, denoised_folder)

# Plot PSNR values
plt.figure(figsize=(10, 5))
plt.plot(psnr_values, marker='o', linestyle='-', color='b')
plt.title('PSNR Values Over Image Pairs')
plt.xlabel('Image Pair Index')
plt.ylabel('PSNR (dB)')
plt.grid(True)
plt.show()

