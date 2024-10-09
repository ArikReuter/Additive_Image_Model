import torch 
from torch import nn
from tqdm import tqdm
from Utils.metrics import var_exp_score, coef_det, mad_explained
from Ablation.Squares.Plots_Squares_Ablation import interpolate_images_gt, interpolate_images, find_centroid_torch, f_img_raw

class SquaresSimulatorPosition:
    """
    Simulate images where the background is black and one white square is placed at a random position. The size of the square is 1/4 of the image size.
    """
    def __init__(self, size, num_images, seed = 42):
        """
        :param size: Size of the images.
        :param num_images: Number of images to simulate.
        """
        self.size = size
        self.num_images = num_images
        self.seed = seed
        torch.manual_seed(self.seed)

    def simulate(self):
        """
        Simulate the images.
        :return: Tensor of shape (num_images, 3, size, size) containing the images and a tensor of shape (num_images,) containing the labels.
        """
        # Create a tensor to hold the images
        images = torch.zeros(self.num_images, 1, self.size, self.size)

        # Size of the white square
        square_size = self.size // 4

        # Generate random quadrants
        quadrants = torch.zeros((self.num_images))

        center_min_x = square_size
        center_max_x = self.size - square_size

        center_max_y = self.size - square_size

        center_min_y = square_size

        center_org_x = self.size //2
        center_org_y = self.size //2

        for i in range(self.num_images):
            center_x = torch.randint(center_min_x, center_max_x, (1,))
            center_y = torch.randint(center_min_y, center_max_y, (1,))

            # assign quadrants such that the upper left quadrant is 0, upper right is 1, lower left is 2, lower right is 3

            if center_x < center_org_x and center_y < center_org_y:
                quadrants[i] = 0
            elif center_x >= center_org_x and center_y < center_org_y:
                quadrants[i] = 1
            elif center_x < center_org_x and center_y >= center_org_y:
                quadrants[i] = 2
            else:
                quadrants[i] = 3

            # Generate the image

            # Set the pixel values of the square to 1
            images[i, 0, center_y - square_size : center_y + square_size, center_x - square_size : center_x + square_size] = 1


        # Convert to RGB by repeating the channel
        images = images.repeat(1, 3, 1, 1)

        return images, quadrants

test_image_simulator = SquaresSimulatorPosition(size = 64, num_images = 1000)
test_images, _ = test_image_simulator.simulate()

def score_prediction_ground_truth(img1,
                                  img2,
                                  images,
                                  prediction_model,
                                  feature_extraction_fun = find_centroid_torch,
                                  prediction_fun = f_img_raw,
                                  n_datapoints = 1000,
                                  n_images= 10,
                                  T_encoding = 2,
                                  T_decoding = 2,
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

  preds_smooth_gt = torch.tensor(preds_smooth_gt)
  preds_smooth_mod = torch.tensor(preds_smooth_mod)


  var_exp = var_exp_score(preds_smooth_mod, preds_smooth_gt)
  mad_exp = mad_explained(preds_smooth_mod, preds_smooth_gt)
  r_score = coef_det(preds_smooth_mod, preds_smooth_gt)

  mse = torch.nn.MSELoss()(preds_smooth_mod, preds_smooth_gt)

  return {
      "var_exp": var_exp.item(),
      "mad_exp": mad_exp.item(),
      "r_score": r_score.item(),
      "mse": mse.item()
  }

def score_model_interpolation(model, n_pairs = 1000, test_images = test_images):
  res_lis = []
  for i in tqdm(list(range(n_pairs))):
    rnd_idx1 = torch.randint(len(test_images), size = (1,1))[0,0].item()
    rnd_idx2 = torch.randint(len(test_images), size = (1,1))[0,0].item()

    img1 = test_images[rnd_idx1]
    img2 = test_images[rnd_idx2]

    res = score_prediction_ground_truth(img1, img2, prediction_model = model)
    res["idx1"] = rnd_idx1
    res["idx2"] = rnd_idx2

    res_lis.append(res)

  import pandas as pd
  res_df = pd.DataFrame(res_lis)

  print(res_df.describe())

  return res_df