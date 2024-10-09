import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

def plot_feature_nn(feature_nn, x_vals_compute, bias, x_vals_to_plot, ax=None, y_data=None, unscale_fun=None, q = 0, device = "cpu"):


    with sns.axes_style('whitegrid'):
      with sns.color_palette('dark'):

        max_fun = lambda x: np.quantile(x, q = 1-q)
        min_fun = lambda x: np.quantile(x, q = q)

        valid_idx = (x_vals_compute >= min_fun(x_vals_compute)) & (x_vals_compute <= max_fun(x_vals_compute))
        print((np.mean(valid_idx)))
        print(len(x_vals_compute))
        print(len(x_vals_to_plot))

        x_vals_compute = x_vals_compute[valid_idx]
        x_vals_to_plot = x_vals_to_plot[valid_idx]
        y_data = y_data[valid_idx]




        min_compute = min(x_vals_compute)
        max_compute = max(x_vals_compute)
        range_compute = np.linspace(min_compute, max_compute, 1000)

        min_plot = min(x_vals_to_plot)
        max_plot = max(x_vals_to_plot)
        range_plot = np.linspace(min_plot, max_plot, 1000)

        feature_nn = feature_nn.eval()
        with torch.no_grad():
            model_input = torch.tensor(range_compute).to(device).reshape(-1, 1).float()
            y_pred = feature_nn(model_input) + bias

            if not unscale_fun is None:
                y_pred = unscale_fun(y_pred)
                y_data = unscale_fun(y_data)

            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 10))
            else:
                fig = plt.gcf()

            fig.set_size_inches(30/2, 10.5/2)
            # Plot the model prediction
            sns.lineplot(x=range_plot, y=y_pred.squeeze().cpu().numpy(), color="r", label="Model Prediction", linewidth=2)

            # Plot the training data
            if y_data is not None:
                sns.scatterplot(x=x_vals_to_plot, y=y_data, color="blue", alpha=0.3, s = 5, label="Training Data", ax=ax)

            # Customize the plot
            ax.set_xlabel("X", fontsize=15)
            ax.set_ylabel("Y", fontsize=15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_linewidth(1.5)
            ax.spines["left"].set_linewidth(1.5)
            ax.tick_params(axis='both', which='major', labelsize=15, width=1.5)
            #ax.set_xticks(fontsize=15)

            # Add a title and legend
            #ax.set_title("Feature Neural Network", fontsize=20)
            ax.legend(fontsize=20, frameon=True)

            # Add a grid
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            return fig, ax

