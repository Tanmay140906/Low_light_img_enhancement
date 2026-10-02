import cv2
import numpy as np
import os
from os import listdir
from matplotlib import pyplot as plt
from skimage import img_as_ubyte, io
from pathlib import Path
import content as opt
import content as ba
import content as tv

input_folder = './Low-images/'
output_folder = './High-images/'


Path(output_folder).mkdir(parents=True, exist_ok=True)

weight_strategy_value = 3  # Weight strategy value for illumination map optimization
brightness_adjustment_value = 0.15  # Brightness adjustment value for gamma correction
sigma_value = 0.001  # Sigma value for NLMeans denoising


def is_valid_image_file(filename):
    """
    Check if the filename has a valid image extension.
    Arguments:
    filename (str): The name of the file.

    Returns:
    bool: True if the file has a valid image extension, False otherwise.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

image_files = []
for file in listdir(input_folder):
    if is_valid_image_file(file):
        image_files.append(file)

for filename in image_files:
    filepath = os.path.join(input_folder, filename)
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    
    # Convert image to RGB and normalize
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Calculate illumination map
    illumination_map = np.max(image_rgb, axis=-1)
    
    # Update illumination map using the optimized function
    optimized_illum_map = opt.optimize_illumination_map(illumination_map, weight_strategy_value)
    
    # Apply brightness adjustment to the illumination map
    brightness_adjusted_illum_map = ba.brightness_adjust(np.abs(optimized_illum_map), brightness_adjustment_value)
    brightness_adjusted_illum_map = np.expand_dims(brightness_adjusted_illum_map, axis=-1)
    
    # Correct image illumination by dividing by the adjusted illumination map
    corrected_image = np.divide(image_rgb, brightness_adjusted_illum_map, where=brightness_adjusted_illum_map != 0)
    
    # Clip image values to the range [0, 1] and convert to float32
    corrected_image = np.clip(corrected_image, 0, 1).astype(np.float32)
    
    # Denoise the corrected image using Total Variation denoising
    denoised_image = tv.denoising_tv(corrected_image, brightness_adjusted_illum_map, weight=sigma_value)

    
    # Plot the original and denoised images with MSE annotation
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(denoised_image)
    axes[1].set_title('Denoised Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save the denoised image
    denoised_image_bgr = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_folder, '_enhanced_' + filename)
    cv2.imwrite(output_path, img_as_ubyte(denoised_image_bgr))
    
print("Denoising complete.")
