# NishaShukla
“Develop a 2D Occupancy  Grid Map of a Room using Overhead Cameras”.
import cv2
import numpy as np

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

    cv2.imshow("Occupancy Grid", grid_visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the overhead camera image
    image_path = "overhead_view.jpg"  # Path to your image
    image = cv2.imread(image_path)

    # Set grid size (number of cells in one dimension)
    grid_size = 10

    # Create the occupancy grid
    occupancy_grid = create_occupancy_grid(image, grid_size)

    # Visualize the occupancy grid
    visualize_grid(occupancy_grid)

    # Print the occupancy grid
    print("Occupancy Grid:\n", occupancy_grid)
