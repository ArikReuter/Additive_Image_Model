from Manipulation_Squares import find_centroid_torch, f_img, plot_images_and_predictions2
import torch 
import numpy as np
from Manipulation_Squares import create_manipulated_image_path


def manipulate_images(img1,
                      images,
                      normal_vec,
                      prediction_model,
                      device = 'cuda',
                      alpha_max = 1.0,
                      feature_extraction_fun = find_centroid_torch,
                      prediction_fun = f_img,
                      n_datapoints = 1000,
                      n_images= 10,
                      T_encoding = 2,
                      T_decoding = 200,
                      ymin=-1.3,
                      ymax=1.3,
                      save_path = None):


  generated_images, cond, intp = create_manipulated_image_path(img1, normal_vec, n_images, alpha_max, T_encoding, T_decoding)

  preds_images = prediction_model(intp.float())[:, 0].detach().cpu().numpy()


  alpha_detailed = (np.linspace(0, alpha_max, n_datapoints, dtype=np.float32))
  intp_detailed = torch.cat([cond +alpha* (torch.norm(cond)/torch.norm(normal_vec))*normal_vec.to(device) for alpha in alpha_detailed])

  preds_detailed = prediction_model(intp_detailed.float())[:, 0].detach().cpu().numpy()
  mean = np.mean(preds_detailed)
  preds_detailed = preds_detailed - mean
  preds_images = preds_images - mean

  feats_gt = [feature_extraction_fun(img.permute(1,2,0))[0] for img in generated_images]
  preds_images_gt = np.array([prediction_fun(feat) for feat in feats_gt])

  preds_images_gt = preds_images_gt - np.mean(preds_images_gt)


  plot_images_and_predictions2(images1 = generated_images,
                               image_preds1 = preds_images_gt,
                               preds_smooth1 = None,
                               save_path = save_path,
                               images2 = None,
                               image_preds2= preds_images,  #preds_images
                               preds_smooth2 = preds_detailed,
                               dpi_save = 400,
                               ymin=-1.3,
                               ymax=1.3)

  return generated_images



