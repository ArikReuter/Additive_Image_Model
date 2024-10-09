import torch 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def find_centroid_torch(img):
    # Convert the image to a binary mask where white pixels are True
    binary_mask = torch.all(img >= 0.8, dim=2)

    # Find the coordinates of the white pixels
    y_coords, x_coords = torch.where(binary_mask)

    # Calculate the centroid
    centroid_x = torch.mean(x_coords.float())
    centroid_y = torch.mean(y_coords.float())

    return int(centroid_x.item()), int(centroid_y.item())


def visualize_original_image_effects(img1, img2, n_steps_images):
    """
    Generate a sequence of images interpolating between img1 and img2 based on the centroid's positions.
    Args:
      img1: first image
      img2: second image
    """

    # Find centroids of the two images
    centroid1 = find_centroid_torch(img1)
    centroid2 = find_centroid_torch(img2)

    # Generate interpolated positions
    interpolated_positions = interpolate_positions(centroid1, centroid2, n_steps_images)

    # Generate and store interpolated images
    interpolated_images = []
    for position in interpolated_positions:
        interpolated_img = generate_img((int(position[0].item()), int(position[1].item())))
        interpolated_images.append(interpolated_img)


    return interpolated_images, interpolated_positions

def generate_img(centroid_position, size=64):
    # Create a black image of the specified size
    img = torch.zeros((size, size, 3), dtype = torch.float)

    # Calculate the top left position of the square
    square_size = 32
    top_left_x = centroid_position[0] - square_size // 2
    top_left_y = centroid_position[1] - square_size // 2

    # Ensure the square doesn't go out of the image boundaries
    top_left_x = max(0, min(size - square_size, top_left_x))
    top_left_y = max(0, min(size - square_size, top_left_y))

    # Draw a white square on the black image
    img[top_left_y:top_left_y + square_size, top_left_x:top_left_x + square_size] = 1

    return img


def interpolate_positions(pos1, pos2, n_steps):
    """
    Linearly interpolate between two positions.
    """
    x1, y1 = pos1
    x2, y2 = pos2

    x_values = torch.linspace(x1, x2 + 1, n_steps)
    y_values = torch.linspace(y1, y2 +1 , n_steps)

    return zip(x_values, y_values)

def plot_images_and_predictions(images, image_preds, preds_smooth, ymin = -1.3, ymax = 1.3):
  """
  Plot the effect of the images
  params:
  images: images to plot
  image_preds: predictions based on images to plot
  preds_smooth: predictions between image_preds to plot
  """
  if image_preds is not None:
    assert len(images) == len(image_preds), "length of images must be equal to length of image_preds"

  with sns.axes_style('darkgrid'):
          with sns.color_palette('dark'):
            fig = plt.figure(figsize=(30, 10.5))
            gs = fig.add_gridspec(2, len(images), height_ratios=[1, 1])

            ax3 = fig.add_subplot(gs[0, :])

            ax3.set_ylim([ymin, ymax])
            if image_preds is not None:
              ax3.plot(range(len(image_preds)), image_preds,'o', markersize=20, c = "blue",linewidth=3)

            x_range_plot_smooth_preds = np.linspace(0, len(images) -1, len(preds_smooth))
            ax3.plot(x_range_plot_smooth_preds, preds_smooth, '-', markersize=10, c = "blue")

            ax3.set_ylabel('Predicted Probability')
            yticks = ax3.get_yticks()


            ax3.set_xticks([]) # remove x_ticks

            # Plot the images
            for i in range(len(images)):
                axi = fig.add_subplot(gs[1, i])
                axi.imshow(images[i].permute(1, 2, 0).cpu())
                axi.xaxis.set_ticklabels([])
                axi.yaxis.set_ticklabels([])

                axi.grid(False)
                axi.set_xlabel(i + 1, fontsize = 20)


            plt.tight_layout()

            plt.show()

