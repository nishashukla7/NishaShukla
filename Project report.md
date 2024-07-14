# Report on Developing a 2D Occupancy Grid Map of a Room Using Overhead Cameras

## 1. Introduction

The purpose of this project is to create a 2D occupancy grid map of a room using images captured by overhead cameras. This type of mapping is crucial for applications in robotics, autonomous navigation, and space utilization analysis. The occupancy grid represents the likelihood of each cell in the grid being occupied by an object.

## 2. Problem Statement

The goal is to develop a Python solution that processes an image from an overhead camera, analyzes the pixel data to determine occupancy, and constructs a grid representation of the room's layout.

## 3. Methodology

### 3.1 Requirements

- **Libraries**: The solution uses the following libraries:
  - OpenCV: For image processing.
  - NumPy: For numerical operations and array manipulation.
  - Matplotlib: For visualizing the occupancy grid.

### 3.2 Steps Involved

1. **Image Input**: Load the overhead camera image.
2. **Image Processing**:
   - Convert the image to grayscale.
   - Apply a binary threshold to distinguish between occupied and free space.
3. **Grid Mapping**:
   - Divide the binary image into a grid.
   - Calculate occupancy for each grid cell based on the number of occupied pixels.
4. **Visualization**: Display the occupancy grid as an image.

## 4. Implementation

### 4.1 Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_occupancy_grid(image, grid_size, threshold=200):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to create a binary image
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)

    # Resize the binary image to the desired grid size
    height, width = binary_image.shape
    grid_height = height // grid_size
    grid_width = width // grid_size
    occupancy_grid = np.zeros((grid_size, grid_size), dtype=int)

    # Populate the occupancy grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Crop the grid cell
            cell = binary_image[i * grid_height:(i + 1) * grid_height,
                                j * grid_width:(j + 1) * grid_width]
            # Calculate occupancy (1 for occupied, 0 for free)
            occupancy_grid[i, j] = 1 if np.sum(cell) > (grid_height * grid_width) / 2 else 0

    return occupancy_grid

def visualize_grid(occupancy_grid):
    # Create a visual representation of the occupancy grid
    grid_visual = np.zeros((occupancy_grid.shape[0] * 10, occupancy_grid.shape[1] * 10), dtype=np.uint8)
    
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            if occupancy_grid[i, j] == 1:
                grid_visual[i*10:(i+1)*10, j*10:(j+1)*10] = 255  # Occupied

    plt.imshow(grid_visual, cmap='gray')
    plt.title("Occupancy Grid")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Load the overhead camera image
    image_path = "overhead_view.jpg"  # Replace with your image file
    image = cv2.imread(image_path)

    # Set grid size (number of cells in one dimension)
    grid_size = 10

    # Create the occupancy grid
    occupancy_grid = create_occupancy_grid(image, grid_size)

    # Visualize the occupancy grid
    visualize_grid(occupancy_grid)

    # Print the occupancy grid
    print("Occupancy Grid:\n", occupancy_grid)
```

### 4.2 Explanation of Key Functions

- **create_occupancy_grid**:
  - Converts the input image to grayscale.
  - Applies a binary threshold to obtain a binary image.
  - Divides the binary image into a grid and assesses the occupancy of each cell.

- **visualize_grid**:
  - Constructs a visual representation of the occupancy grid and displays it using Matplotlib.

## 5. Running the Code in Google Colab

### 5.1 Steps

1. **Open Google Colab**:
   - Navigate to [Google Colab](https://colab.research.google.com/).

2. **Create a New Notebook**:
   - Click on “File” > “New Notebook”.

3. **Install Required Libraries**:
   - In the first cell, run the following command:
   ```python
   !pip install opencv-python matplotlib
   ```

4. **Upload the Image**:
   - Use the following code to upload your overhead camera image:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

5. **Copy and Paste the Solution**:
   - Copy the entire code from the implementation section and paste it into a new cell.

6. **Run the Code**:
   - Make sure to update the line `image_path = "overhead_view.jpg"` with the name of the uploaded image file.
   - Run the cell containing the code. The occupancy grid will be generated and displayed.

7. **View Output**:
   - The occupancy grid will be shown as a visual output, and the grid values will be printed in the console.

## 6. Conclusion

This project successfully developed a Python solution for generating a 2D occupancy grid map from an overhead camera image. The method demonstrated how image processing techniques can be applied to interpret spatial data for various applications. Future work could involve integrating advanced object detection techniques and refining the grid resolution based on the environment's complexity.
