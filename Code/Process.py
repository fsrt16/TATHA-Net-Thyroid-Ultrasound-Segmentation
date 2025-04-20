import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from skimage.io import imread
from tqdm import tqdm

class ThyroidUltrasoundSegmentation:
    """
    A class to preprocess and manage thyroid ultrasound segmentation data.
    This class provides methods for:
    - Loading images and masks from directories
    - Creating a CSV file mapping images to masks
    - Loading datasets into numpy arrays for model training
    - Visualizing the images and their corresponding masks

    Attributes:
    -----------
    image_dir : str
        Path to the directory containing the input ultrasound images.
    mask_dir : str
        Path to the directory containing the corresponding mask images.
    output_csv : str
        Path to the CSV file where image-mask mappings are saved.
    """
    
    def __init__(self, image_dir, mask_dir, output_csv):
        """
        Initializes the class with the given directories for images and masks, and the CSV output path.

        Parameters:
        -----------
        image_dir : str
            Path to the directory containing the input ultrasound images.
        mask_dir : str
            Path to the directory containing the corresponding mask images.
        output_csv : str
            Path to the CSV file where image-mask mappings are saved.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_csv = output_csv
    
    def ld_dimcon(self, path, is_mask=False):
        """
        Loads an image from a given path and returns it. If the image is a mask, it binarizes the mask.

        Parameters:
        -----------
        path : str
            Path to the image file.
        is_mask : bool
            Whether the image is a mask. If True, the image is binarized.

        Returns:
        --------
        np.ndarray
            The processed image as a numpy array.
        """
        # Read the image in grayscale
        image = imread(path, as_gray=True)
        
        # Ensure the image has two dimensions (height x width)
        if image.ndim != 2:
            raise ValueError(f"Image at {path} has unexpected dimensions: {image.shape}")
        
        # If it's a mask, binarize it (convert all pixels > 0.5 to 1)
        if is_mask:
            image = (image > 0.5).astype(np.uint8)

        return image
    
    def create_mapping_csv(self):
        """
        Creates a CSV file that maps image filenames to mask filenames based on numeric identifiers.

        This function assumes the image and mask filenames are numeric and share the same base name.
        It saves the mapping to a CSV file.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the image-mask mapping.
        """
        # Extract filenames from the directories
        image_files = sorted(os.listdir(self.image_dir))
        mask_files = sorted(os.listdir(self.mask_dir))
        
        # Helper function to extract numeric identifiers from filenames
        def extract_numeric(filename):
            return int(os.path.splitext(filename)[0])  # Extract number before '.PNG'
        
        # Create dictionaries for image-mask mapping
        image_dict = {extract_numeric(img): img for img in image_files if img.endswith('.PNG')}
        mask_dict = {extract_numeric(mask): mask for mask in mask_files if mask.endswith('.PNG')}
        
        # Find common keys between image and mask dictionaries
        common_keys = set(image_dict.keys()).intersection(mask_dict.keys())
        mapped_data = [{"Image": image_dict[key], "Mask": mask_dict[key]} for key in common_keys]
        
        # Save the mapping to a CSV file
        df = pd.DataFrame(mapped_data)
        df.to_csv(self.output_csv, index=False)
        print(f"Mapping CSV created at: {self.output_csv}")
        return df
    
    def load_dataset_from_csv(self, num_images=100):
        """
        Loads a dataset of images and masks into numpy arrays based on a CSV mapping.

        Parameters:
        -----------
        num_images : int
            The number of images to load (if there are more than this, a random subset will be chosen).

        Returns:
        --------
        tuple
            A tuple containing:
            - X : np.ndarray (input images)
            - Y : np.ndarray (corresponding masks)
        """
        # Load the CSV file that contains the image-mask mapping
        df = pd.read_csv(self.output_csv)
        X, Y = [], []
        
        # Loop through each row in the mapping CSV
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(self.image_dir, row['Image'])
            mask_path = os.path.join(self.mask_dir, row['Mask'])
            
            # Load the image and the corresponding mask
            img = self.ld_dimcon(image_path, is_mask=False)
            mask = self.ld_dimcon(mask_path, is_mask=True)
            
            # Add image and mask to the list
            X.append(img)
            Y.append(mask)
        
        # Convert the lists to numpy arrays
        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='float32')
        
        print(f'Dataset shapes: {X.shape}, {Y.shape}')
        
        # If there are more images than requested, randomly sample
        if len(X) > num_images:
            indices = random.sample(range(len(X)), num_images)
            X, Y = X[indices], Y[indices]
        
        return X, Y
    
    def visualize_sample(self, ind, X, Y):
        """
        Visualizes a sample image and its corresponding mask.

        Parameters:
        -----------
        ind : int
            The index of the image-mask pair to visualize.
        X : np.ndarray
            Array of input images.
        Y : np.ndarray
            Array of corresponding masks.
        """
        plt.figure(figsize=(12, 6))

        # Input Image
        plt.subplot(1, 2, 1)
        plt.imshow(X[ind], cmap='gray')
        plt.title('Input Image')

        # Mask
        plt.subplot(1, 2, 2)
        plt.imshow(Y[ind], cmap='gray')
        plt.title('Mask')

        plt.show()


# Example usage:
if __name__ == '__main__':
    # Define paths
    base_dir = '/kaggle/input/thyroidultrasound'
    image_dir = os.path.join(base_dir, 'p_image')  # Images directory
    mask_dir = os.path.join(base_dir, 'p_mask')    # Masks directory
    output_csv = 'image_mask_mapping.csv'

    # Create an instance of the ThyroidUltrasoundSegmentation class
    processor = ThyroidUltrasoundSegmentation(image_dir, mask_dir, output_csv)

    # Create the CSV mapping of images to masks
    df = processor.create_mapping_csv()

    # Load the dataset using the created CSV mapping
    X, Y = processor.load_dataset_from_csv(num_images=100)

    # Visualize the first 10 samples in the dataset
    for ind in range(min(10, len(X))):
        processor.visualize_sample(ind, X, Y)
