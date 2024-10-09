import torch

def generate_img(r, g=0.5, b=0.5, size=64):
    """
    Generate an image of shape (3, size, size) where all pixels are colored according to r, g, b.

    Args:
        r (float): Red channel intensity (between 0 and 1).
        g (float): Green channel intensity (between 0 and 1).
        b (float): Blue channel intensity (between 0 and 1).
        size (int): Size of the image (width and height).

    Returns:
        torch.Tensor: Generated image tensor with shape (3, size, size).
    """
    if not (0 <= r <= 1) or not (0 <= g <= 1) or not (0 <= b <= 1):
        raise ValueError("RGB values should be between 0 and 1")

    # Create an empty image tensor
    image = torch.zeros(3, size, size)

    # Fill the image tensor with the specified RGB values
    image[0, :, :] = r
    image[1, :, :] = g
    image[2, :, :] = b

    return image


def interpolate_images_linear_torch(img1, img2, n_images=10):
    # Ensure the input images are PyTorch tensors
    if not (isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor)):
        raise TypeError("img1 and img2 must be PyTorch tensors")

    # Calculate the average RGB values for each image
    r1, g1, b1 = img1.mean(dim=[1, 2])
    r2, g2, b2 = img2.mean(dim=[1, 2])

    # Create a sequence of interpolated RGB values
    r_values = torch.linspace(r1, r2, steps=n_images)
    g_values = torch.linspace(g1, g2, steps=n_images)
    b_values = torch.linspace(b1, b2, steps=n_images)

    # Generate and store the interpolated images
    interpolated_images = []
    for r, g, b in zip(r_values, g_values, b_values):
        # Create an image with the interpolated color
        img = torch.full_like(img1, 0)  # Initialize with zeros
        img[0, :, :] = r  # Red channel
        img[1, :, :] = g  # Green channel
        img[2, :, :] = b  # Blue channel
        interpolated_images.append(img)

    return torch.stack(interpolated_images)

def average_red_value_from_tensor(image_tensor):
    # Extract the Red channel (channel 0)
    red_channel = image_tensor[0, :, :]

    # Calculate the average Red value
    average_red = torch.mean(red_channel)

    return average_red.item()

def average_rgb_vals_from_tensor(image_tensor):
    """
    Calculate the average RGB values from a given image tensor.

    Args:
        image_tensor (torch.Tensor): Input image tensor with shape (3, height, width).

    Returns:
        tuple: A tuple containing the average values for Red, Green, and Blue channels.
    """
    if image_tensor.shape[0] != 3:
        raise ValueError("Input tensor should have shape (3, height, width) for RGB image")

    # Calculate the average values for each RGB channel
    average_red = torch.mean(image_tensor[0, :, :])
    average_green = torch.mean(image_tensor[1, :, :])
    average_blue = torch.mean(image_tensor[2, :, :])

    return (average_red.item(), average_green.item(), average_blue.item())

def average_red_value_batch(image_batch):
    # Extract the Red channels (channel 0) for all images in the batch
    red_channels = image_batch[:, 0, :, :]

    # Calculate the average Red value for each image in the batch
    average_red_values = torch.mean(red_channels, dim=(1, 2))

    return average_red_values