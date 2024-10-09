import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from os import listdir
from os.path import isfile, join

from Utils.dataloaders import ImageDataset

class ImageInterpolator:
    def __init__(self, data_path, pretrained_encoder, model, device, image_size, save_dir):
        self.data_path = data_path
        self.pretrained_encoder = pretrained_encoder
        self.model = model
        self.device = device
        self.image_size = image_size
        self.save_dir = save_dir

        # Load dataset
        self.data = ImageDataset(
            data_path,
            image_size=self.image_size,
            exts=['jpg', 'JPG', 'png'],
            do_augment=False
        )

        # Map filenames to indices
        self.files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        self.fileNum2fileIdx = {
            int(name.replace(".jpg", "")): i for i, name in enumerate(self.files)
        }

    def select_images(self, i_idx, j_idx):
        batch = torch.stack([
            self.data[i_idx]['img'],
            self.data[j_idx]['img'],
        ])
        return batch

    def interpolate(self, batch, n_interpolation_steps=6, T=250, T_render=200):
        cond = self.pretrained_encoder.encode(batch.to(self.device))
        xT = self.pretrained_encoder.encode_stochastic(batch.to(self.device), cond, T=T)
        alpha = torch.linspace(0, 1, steps=n_interpolation_steps, device=self.device)
        intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]

        # Spherical interpolation function
        def cos(a, b):
            a = a.view(-1)
            b = b.view(-1)
            a = F.normalize(a, dim=0)
            b = F.normalize(b, dim=0)
            return (a * b).sum()

        theta = torch.arccos(cos(xT[0], xT[1]))
        x_shape = xT[0].shape
        xT0_flat = xT[0].flatten(0, 2)
        xT1_flat = xT[1].flatten(0, 2)
        intp_x = (
            torch.sin((1 - alpha[:, None]) * theta) * xT0_flat[None] +
            torch.sin(alpha[:, None] * theta) * xT1_flat[None]
        ) / torch.sin(theta)
        intp_x = intp_x.view(-1, *x_shape)

        pred = self.pretrained_encoder.render(intp_x, intp, T=T_render)
        return pred, intp

    @staticmethod
    def find_nearest_multiples(x):
        """Find the nearest greater and smaller multiples of 500 to a given float."""
        greater_multiple = (int(x) // 500 + 1) * 500
        smaller_multiple = (int(x) // 500) * 500
        return greater_multiple, smaller_multiple

    def unstandardize_and_exp_targets(self, x):
        """
        Placeholder function to reverse any standardization and exponentiate the targets.
        Replace with the actual transformation used during data preprocessing.
        """
        # Example: If targets were standardized and then logarithm was taken
        # x = (x * self.std) + self.mean
        # x = torch.exp(x)
        return x

    def plot_results(self, pred, intp, i_idx, j_idx, n_interpolation_steps=6):
        # Generate interpolated predictions
        n_interpolation_steps_pred = 100 * n_interpolation_steps
        alpha_pred = torch.linspace(0, 1, steps=n_interpolation_steps_pred, device=self.device)
        intp_pred = (
            intp[0][None] * (1 - alpha_pred[:, None]) +
            intp[1][None] * alpha_pred[:, None]
        )

        preds_images = self.unstandardize_and_exp_targets(
            self.model.mlp(intp_pred).detach().cpu()
        ).numpy()

        # Plot settings
        sns.set_style('darkgrid')
        sns.set_palette('dark')

        fig = plt.figure(figsize=(30, 10.5))
        gs = fig.add_gridspec(2, n_interpolation_steps, height_ratios=[1, 1])

        # Plot the predictions
        ax3 = fig.add_subplot(gs[0, :])
        ax3.plot(range(len(intp_pred)), preds_images, '-', markersize=20, c="blue", linewidth=3)

        x_red_plot = np.linspace(0, n_interpolation_steps_pred, n_interpolation_steps)
        img_red_plot = self.unstandardize_and_exp_targets(
            self.model.mlp(intp).detach().cpu()
        ).numpy()
        ax3.plot(x_red_plot, img_red_plot, 'o', markersize=15, c="blue")

        y_min_r = self.find_nearest_multiples(preds_images.min())[1]
        y_max_r = self.find_nearest_multiples(preds_images.max())[0]
        ax3.set_ylim([y_min_r, y_max_r])

        ax3.set_ylabel('Price in ZAR', fontsize=30)
        yticks = ax3.get_yticks()
        yticklabels = ['{:,.0f}'.format(y) + ' ZAR' for y in yticks]
        ax3.set_yticklabels(yticklabels, fontsize=30)
        ax3.set_xticks([])

        # Plot the images
        for i in range(n_interpolation_steps):
            axi = fig.add_subplot(gs[1, i])
            axi.imshow(pred[i].permute(1, 2, 0).cpu())
            axi.set_xticks([])
            axi.set_yticks([])

        plt.tight_layout()
        title = f"Interpolation between indices {i_idx} and {j_idx}"
        ax3.set_title(title, fontsize=40)

        # Save and display the plot
        plt.savefig(
            f'{self.save_dir}/interpolation_{i_idx}_{j_idx}.png',
            bbox_inches='tight'
        )
        plt.show()

    def run_interpolations(self, idx_list):
        for i_idx, j_idx in idx_list:
            batch = self.select_images(i_idx, j_idx)
            pred, intp = self.interpolate(batch)
            self.plot_results(pred, intp, i_idx, j_idx)

# Usage Example:

# Initialize the required components
# data_path = 'path_to_images_directory'
# pretrained_encoder = ...  # Your pretrained encoder model
# model = ...               # Your model with 'mlp' attribute
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# image_size = 128          # Example image size
# save_dir = 'path_to_save_plots'

# Create an instance of the ImageInterpolator
# interpolator = ImageInterpolator(data_path, pretrained_encoder, model, device, image_size, save_dir)

# Define the list of index pairs for interpolation
# idx_list = [
#     (0, 4),
#     (3, 80),
#     (4, 53),
#     (5, 26),
#     # Add more pairs as needed
# ]

# Run the interpolation and plotting
# interpolator.run_interpolations(idx_list)
