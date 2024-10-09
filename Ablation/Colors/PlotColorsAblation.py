import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import time



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

def plot_images_and_predictions(images, image_preds, preds_smooth):
    """
    Plot the effect of the images
    params:
    images: images to plot
    image_preds: predictions based on images to plot
    preds_smooth: predictions between image_preds to plot
    """

    assert len(images) == len(image_preds), "length of images must be equal to length of image_preds"

    # Calculate r_vals as the mean of the image predictions
    r_vals = torch.mean(images[:, 0], dim=(-1, -2))
    print(r_vals.shape)

    with sns.axes_style('darkgrid'):
        with sns.color_palette('dark'):
            fig = plt.figure(figsize=(30, 15))
            gs = fig.add_gridspec(2, len(images), height_ratios=[1, 1], wspace = 0.0)

            ax3 = fig.add_subplot(gs[0, :])
            ax3.plot(range(len(image_preds)), image_preds, 'o', markersize=20, c="blue", linewidth=3)

            x_range_plot_smooth_preds = np.linspace(0, len(image_preds) - 1, len(preds_smooth))
            ax3.plot(x_range_plot_smooth_preds, preds_smooth, '-', markersize=10, c="blue")

            ax3.set_ylabel('Predicted Probability')

            # Set the x-tick labels to r_vals
            ax3.set_xticks(range(len(r_vals)))
            ax3.set_xticklabels([f'{val:.2f}' for val in r_vals])

            # Plot the images
            for i in range(len(images)):
                axi = fig.add_subplot(gs[1, i])
                axi.imshow(images[i].permute(1, 2, 0).cpu())
                #axi.grid(True, linestyle='--', color='black', linewidth=3)
                axi.set_xticks([images[i].shape[1] / 2])
                axi.set_yticks([images[i].shape[1] / 2])
                axi.xaxis.set_ticklabels([])
                axi.yaxis.set_ticklabels([])
                axi.set_xlabel(f'R: {r_vals[i]:.2f}', fontsize=20)  # Set the xlabel to the corresponding r_val
                axi.grid(False)
                axi.margins(0)

            plt.tight_layout(pad=0.5)
            plt.show()


def plot_images_and_predictions2(images1, image_preds1, preds_smooth1,
                                images2, image_preds2, preds_smooth2, save_path = None, dpi_save = 400,
                                ymin=-1.3, ymax=1.3):
    """
    Plot the effect of two sets of images and their predictions.
    params:
    images1, images2: images to plot
    image_preds1, image_preds2: predictions based on images to plot  GROUND TRUTH
    preds_smooth1, preds_smooth2: predictions between image_preds to plot PREDICTIONS
    """
    # Assert that the lengths of images and image_preds are equal for both sets
    if image_preds1 is not None and images1 is not None:
        assert len(images1) == len(image_preds1), "Length of images1 must be equal to length of image_preds1"
    if image_preds2 is not None and images2 is not None:
        assert len(images2) == len(image_preds2), "Length of images2 must be equal to length of image_preds2"

    with sns.axes_style('darkgrid'):
        with sns.color_palette('dark'):
            # Adjust figure size to accommodate two sets of images
            fig = plt.figure(figsize=(23/2, 10/2), constrained_layout=False, dpi = dpi_save)
            # Create a grid with 4 rows: 1 for plot, 3 for images

            if images2 is not None:
              gs = fig.add_gridspec(3, max(len(images1), len(images2)), height_ratios=[1, 0.5, 0.5], hspace = 0.2, wspace=0.00) #the height ratios are a kinda a hack to cicumvent some weird vertical padding padding
            else:
              gs = fig.add_gridspec(3, len(images1), height_ratios=[1, 0.5, 0.5], hspace = 0.05, wspace=0.05) #the height ratios are a kinda a hack to cicumvent some weird vertical padding padding

            # Plot for predictions
            ax3 = fig.add_subplot(gs[0, :])
            ax3.set_ylim([ymin, ymax])


            if preds_smooth1 is not None:

              x_range_plot_smooth_preds1 = np.linspace(0, len(images1) - 1, len(preds_smooth1))
              print(len(x_range_plot_smooth_preds1), len(preds_smooth1))
              #ax3.scatter(x_range_plot_smooth_preds1, preds_smooth1, c="blue")
              ax3.plot(x_range_plot_smooth_preds1, preds_smooth1, '-', markersize=10, c="blue")


            if preds_smooth2 is not None:
              x_range_plot_smooth_preds2 = np.linspace(0, len(images1) - 1, len(preds_smooth2))
              #ax3.scatter(x_range_plot_smooth_preds2, preds_smooth1, c = "blue")
              print(len(x_range_plot_smooth_preds2), len(preds_smooth2))
              ax3.plot(x_range_plot_smooth_preds2, preds_smooth2, '-', markersize=10, c="red")

            # Plot first set of predictions
            if image_preds1 is not None:
                ax3.plot(range(len(image_preds1)), image_preds1, 'o', markersize=5, c="blue", linewidth=3, label = "Reference")

            # Plot second set of predictions
            if image_preds2 is not None:
                ax3.plot(range(len(image_preds2)), image_preds2, 'o', markersize=5, c="red", linewidth=3, label = "Prediction")



            ax3.set_ylabel('Response')
            ax3.set_xticks([])  # remove x_ticks
            ax3.legend()

            # Plot the first set of images

            if images1 is not None:
              axes1_list = []
              for i in range(len(images1)):
                  axi = fig.add_subplot(gs[1, i])
                  axi.imshow(images1[i].permute(1,2,0).cpu(), aspect = "equal")
                  axi.xaxis.set_ticklabels([])
                  axi.yaxis.set_ticklabels([])
                  axi.grid(False)
                  axes1_list.append(axi)
                  r_value = average_red_value_from_tensor(images1[i])
                  #axi.set_xlabel("{:.2f}".format(r_value), fontsize = 10)

            # Plot the second set of images
            if images2 is not None:
              axes2_list = []
              for i in range(len(images2)):
                  axi = fig.add_subplot(gs[2, i])
                  axi.imshow(images2[i].permute(1,2,0).cpu(), aspect = "equal")
                  axi.xaxis.set_ticklabels([])
                  axi.yaxis.set_ticklabels([])
                  axi.grid(False)
                  axes2_list.append(axi)
                  r_value = average_red_value_from_tensor(images2[i])
                  #axi.set_xlabel("{:.2f}".format(r_value), fontsize = 10)

            if images1 is not None and images2 is not None:
              axes1_list[0].annotate('Reference',
              xy=(0.0, 0.307), xycoords='figure fraction',
              horizontalalignment='left', verticalalignment='center')
              axes2_list[0].annotate('Generated Images',
              xy=(0.0, 0.12), xycoords='figure fraction',
              horizontalalignment='left', verticalalignment='center')

            if images2 is None and images1 is not None:
              axes1_list[0].annotate('Generated Images',
              xy=(0.0, 0.22), xycoords='figure fraction',
              horizontalalignment='left', verticalalignment='center')

            #fig.subplots_adjust(hspace = 0.0)  # Adjust this value to control the space
            #fig.tight_layout()


            if save_path is not None:
              timestr = time.strftime("%Y%m%d-%H%M%S")

              file_name = f"GT_and_Model_plot_{timestr}.png"
              fig.savefig(f"{save_path}/{file_name}", dpi = dpi_save, bbox_inches='tight')

            plt.show()

def interpolate_quadrants(q1, q2, model, prediction_model, labels, images, device = "cuda" if torch.cuda.is_available() else "cpu",
                          T_encoding = 2, T_decoding = 250, n_datapoints = 1000, n_images = 20):
  """
  Interpolates between images where the white square is in quadrant q1 and quadrant q2.
  params:
    q1: quadrant for first image
    q2: quadrant for last image
    T_encoding: number of steps for the stochastic encoder
    T_decoding: number of steps for the stochastic decoder
  """

  img1 = images[:999][labels[:999] == q1][0]
  img2 = images[:999][labels[:999] == q2][0]

  batch = torch.stack([
    img1,
    img2,
  ])

  cond = model.encode(batch.to(device))
  xT = model.encode_stochastic(batch.to(device), cond, T=T_encoding)

  alpha = torch.tensor(np.linspace(0, 1, n_images, dtype=np.float32)).to(cond.device)
  intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]

  def cos(a, b):
      a = a.view(-1)
      b = b.view(-1)
      a = F.normalize(a, dim=0)
      b = F.normalize(b, dim=0)
      return (a * b).sum()

  theta = torch.arccos(cos(xT[0], xT[1]))
  x_shape = xT[0].shape
  intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None] + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]) / torch.sin(theta)
  intp_x = intp_x.view(-1, *x_shape)

  images = model.render(intp_x, intp, T=T_decoding)


  preds_images = prediction_model(intp)[:, 0].detach().cpu().numpy()

  alpha_detailed = torch.tensor(np.linspace(0, 1, n_datapoints, dtype=np.float32)).to(cond.device)
  intp_detailed = cond[0][None] * (1 - alpha_detailed[:, None]) + cond[1][None] * alpha_detailed[:, None]

  preds_detailed = prediction_model(intp)[:, 0].detach().cpu().numpy()

  plot_images_and_predictions(images, preds_images,preds_detailed)



def interpolate_images(img1, img2, prediction_model, model, images, device = "cuda" if torch.cuda.is_available() else "cpu",
                       T_encoding = 2, T_decoding = 200, n_datapoints = 1000, n_images = 10):
  """
  Interpolates between images where the white square is in quadrant q1 and quadrant q2.
  params:
    img1: first image
    img2: second image
    T_encoding: number of steps for the stochastic encoder
    T_decoding: number of steps for the stochastic decoder
  """

  batch = torch.stack([
    img1,
    img2,
  ])

  cond = model.encode(batch.to(device))
  xT = model.encode_stochastic(batch.to(device), cond, T=T_encoding)

  alpha = torch.tensor(np.linspace(0, 1, n_images, dtype=np.float32)).to(cond.device)
  intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]

  def cos(a, b):
      a = a.view(-1)
      b = b.view(-1)
      a = F.normalize(a, dim=0)
      b = F.normalize(b, dim=0)
      return (a * b).sum()

  theta = torch.arccos(cos(xT[0], xT[1]))
  x_shape = xT[0].shape
  intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None] + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]) / torch.sin(theta)
  intp_x = intp_x.view(-1, *x_shape)

  images = model.render(intp_x, intp, T=T_decoding)


  preds_images = prediction_model(intp)[:, 0].detach().cpu().numpy()
  #preds_images = preds_images - np.mean(preds_images)


  alpha_detailed = torch.tensor(np.linspace(0, 1, n_datapoints, dtype=np.float32)).to(cond.device)
  intp_detailed = cond[0][None] * (1 - alpha_detailed[:, None]) + cond[1][None] * alpha_detailed[:, None]

  preds_detailed = prediction_model(intp_detailed)[:, 0].detach().cpu().numpy()

  mean = np.mean(preds_detailed)

  preds_detailed = preds_detailed - mean
  preds_images = preds_images - mean

  #plot_images_and_predictions(images, preds_images,preds_detailed)

  return images, preds_images,preds_detailed


def interpolate_images_gt(img1, img2, r_vals, feature_extraction_fun = average_red_value_from_tensor, prediction_fun = lambda x: torch.sin(2*torch.pi*x), n_datapoints = 1000, n_images= 10):
  #if img1.shape[-1] != 3:
  #  img1 = img1.permute(1, 2, 0)
   # img2 = img2.permute(1,2,0)

  min_xvals = torch.min(r_vals)
  max_xvals = torch.max(r_vals)
   
  def standardize_x_vals(x):
      return (x - min_xvals) / (max_xvals - min_xvals)   #scale position values between 0 and 1

  interpolated_images= interpolate_images_linear_torch(img1, img2, n_images)
  interpolated_images = [img/2 +0.5 for img in interpolated_images]

  latent_1 = standardize_x_vals(feature_extraction_fun(img1))
  latent_2 = standardize_x_vals(feature_extraction_fun(img2))

  print(latent_1, latent_2)



  latent_intp = np.linspace(latent_1, latent_2, n_datapoints)
  #latent_intp = (latent_intp - np.min(latent_intp))/(np.max(latent_intp) - np.min(latent_intp))

  preds_smooth = np.array([prediction_fun(latent_i).item() for latent_i in latent_intp])

  interval = n_datapoints // (n_images - 1)

  # Generate the indices of the selected points
  selected_indices = [i * interval for i in range(n_images - 1)] + [n_datapoints - 1]

  preds_images = preds_smooth[selected_indices]

  mean = np.mean(preds_smooth)
  preds_smooth = preds_smooth - mean
  preds_images = preds_images -mean

  #interpolated_images = torch.stack(interpolated_images)

  #plot_images_and_predictions(interpolated_images, image_preds = preds_images, preds_smooth = preds_smooth)

  return interpolated_images, preds_images, preds_smooth



def interpolate_images_gt_and_model(img1,
                                    img2,
                                    prediction_model,
                                    prediction_fun,
                                    images,
                                    feature_extraction_fun = average_red_value_from_tensor,
                                    n_datapoints = 1000,
                                    n_images= 10,
                                    T_encoding = 2,
                                    T_decoding = 200,
                                    ymin=-1.3,
                                    ymax=1.3,
                                    save_path = None
                                    ):
  interpolated_images_gt, preds_images_gt, preds_smooth_gt = interpolate_images_gt(
      img1,
      img2,
      feature_extraction_fun = feature_extraction_fun,
      prediction_fun = prediction_fun,
      images = images,
      n_datapoints = n_datapoints,
      n_images = n_images
  )



  interpolated_images_mod, preds_images_mod, preds_smooth_mod = interpolate_images(img1,
                                 img2,
                                 prediction_model,
                                 images = images,
                                 T_encoding = T_encoding,
                                 T_decoding = T_decoding,
                                 n_datapoints = n_datapoints,
                                 n_images = n_images)


  plot_images_and_predictions2(images1 = interpolated_images_gt,
                               image_preds1 = preds_images_gt,
                               preds_smooth1 = preds_smooth_gt,
                               images2 =interpolated_images_mod ,
                               image_preds2 = preds_images_mod,
                               preds_smooth2 = preds_smooth_mod,
                               ymin=ymin,
                               ymax=ymax,
                               save_path = save_path
                               )