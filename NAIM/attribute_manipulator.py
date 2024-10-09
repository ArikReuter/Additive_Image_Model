import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class AttributeManipulator:
    def __init__(self, cls_conf, cls_model, pretrained_encoder, data, device, fileNum2fileIdx, CelebAttrDataset, model):
        self.cls_conf = cls_conf
        self.cls_model = cls_model
        self.pretrained_encoder = pretrained_encoder
        self.data = data
        self.device = device
        self.fileNum2fileIdx = fileNum2fileIdx
        self.CelebAttrDataset = CelebAttrDataset
        self.model = model

    def setup_model(self):
        state = torch.load(self.cls_conf.pretrain.path, map_location='cpu')
        print('latent step:', state['global_step'])
        self.cls_model.load_state_dict(state['state_dict'], strict=False)
        self.cls_model.to(self.device)

    def manipulate_attribute(self, i_man_lis, attr_lis, n_manipulation_steps=10):
        for i_m_idx in tqdm(i_man_lis):
            i_idx = self.fileNum2fileIdx[i_m_idx]
            batch = torch.stack([self.data[i_idx]['img']])
            plt.imshow(batch[0].permute([1, 2, 0]) / 2 + 0.5)
            plt.show()

            cond = self.pretrained_encoder.encode(batch.to(self.device))
            xT = self.pretrained_encoder.encode_stochastic(batch.to(self.device), cond, T=50)

            for attribute_to_change in attr_lis:
                print(f'Manipulating: {attribute_to_change}')
                for direction in ["Plus", "Minus"]:
                    sgn_dir = -1 if direction == "Minus" else 1
                    cls_id = self.CelebAttrDataset.cls_to_id[attribute_to_change]
                    cond2_org = self.cls_model.normalize(cond)

                    step_range = sgn_dir * np.linspace(0, 0.4, num=n_manipulation_steps)
                    cond2_range = [cond2_org + step_size * math.sqrt(512) * F.normalize(
                        self.cls_model.classifier.weight[cls_id][None, :], dim=1) for step_size in step_range]
                    cond2_lis = [self.cls_model.denormalize(cond_final) for cond_final in cond2_range]

                    intp_x = xT.repeat(len(cond2_lis), 1, 1, 1)
                    images = self.pretrained_encoder.render(intp_x, torch.cat(cond2_lis), T=200)

                    self._plot_images(images, attribute_to_change, direction, i_idx)

    def _plot_images(self, images, attribute_to_change, direction, i_idx):
        fig, ax1 = plt.subplots(1, len(images), figsize=(5 * 10, 5))
        for i in range(len(images)):
            ax1[i].imshow(images[i].permute(1, 2, 0).cpu())
            ax1[i].set_xlabel(f"Index: {i}")
        plt.tight_layout()
        title = f"Manipulation of Attribute \"{attribute_to_change}\" - {direction}"
        plt.suptitle(title, fontsize=20)
        plt.show()

        # Save the figure
        fig.savefig(f'save_path/Plots/Manipulation_{i_idx}_{attribute_to_change}_{direction}.png')

    def latent_interpolation(self, i_idx, attr_lis, direction="Plus"):
        batch = torch.stack([self.data[i_idx]['img']])
        plt.imshow(batch[0].permute([1, 2, 0]) / 2 + 0.5)
        plt.show()

        cond = self.pretrained_encoder.encode(batch.to(self.device))
        xT = self.pretrained_encoder.encode_stochastic(batch.to(self.device), cond, T=50)

        for attribute_to_change in attr_lis:
            print(f"Interpolating for: {attribute_to_change}")
            sgn_dir = -1 if direction == "Minus" else 1
            cls_id = self.CelebAttrDataset.cls_to_id[attribute_to_change]

            cond2_org = self.cls_model.normalize(cond)
            n_manipulation_steps = 6
            step_range = sgn_dir * np.linspace(0, 0.4, num=n_manipulation_steps)

            cond2_range = [cond2_org + step_size * math.sqrt(512) * F.normalize(
                self.cls_model.classifier.weight[cls_id][None, :], dim=1) for step_size in step_range]
            cond2_lis = [self.cls_model.denormalize(cond_final) for cond_final in cond2_range]

            images = self.pretrained_encoder.render(xT.repeat(len(cond2_lis), 1, 1, 1), torch.cat(cond2_lis), T=200)

            self._plot_images(images, attribute_to_change, direction, i_idx)

    def plot_latent_interpolation(self, cond2_lis_pred, alpha_pred):
        fig = plt.figure()
        plt.scatter(range(len(alpha_pred)), self.model.mlp(torch.stack(cond2_lis_pred).squeeze(1)).detach().cpu().numpy())
        plt.title("Latent Interpolation")
        plt.show()

    def find_nearest_multiples(self, x):
        greater_multiple = (int(x) // 500 + 1) * 500
        smaller_multiple = (int(x) // 500) * 500
        return greater_multiple, smaller_multiple


# Usage example:
# Initialize necessary objects and pass them into the AttributeManipulator class

#cls_conf = ffhq256_autoenc_cls()
#cls_conf.pretrain.path = "/path/to/last.ckpt"
#cls_conf.latent_infer_path = "/path/to/latent.pkl"

#cls_model = ClsModel(cls_conf)
#pretrained_encoder = ...  # Your pretrained encoder
#data = ...  # Your dataset
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#fileNum2fileIdx = ...  # File to index mapping
#CelebAttrDataset = ...  # Your CelebA dataset
#model = ...  # Your trained model

# Create an instance of the manipulator
#manipulator = AttributeManipulator(cls_conf, cls_model, pretrained_encoder, data, device, fileNum2fileIdx, CelebAttrDataset, model)

# Manipulate attributes
#i_man_lis = [88, 75, 249, 764, 784, 97, 880, 966, 761]
#attr_lis = ['Attractive', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Male', 'Young']
#manipulator.manipulate_attribute(i_man_lis, attr_lis)

# Perform latent interpolation
#i_idx = 54
#manipulator.latent_interpolation(i_idx, attr_lis)
