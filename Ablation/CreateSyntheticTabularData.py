import numpy as np
import matplotlib.pyplot as plt

class SyntheticDataGenerator:
    def __init__(self, function_list, noise_variance=1, n_datapoints=100, x_min=0, x_max=1, equally_spaced=True):
        """
        Initialize the SyntheticDataGenerator with the necessary parameters.

        Params:
            function_list: List of functions [f_1, f_2, ..., f_k] to use for the response
            noise_variance: The variance of the added noise
            n_datapoints: Number of samples to generate
            x_min: Minimum x_value for each covariate
            x_max: Maximum x_value for each covariate
            equally_spaced: If True, use equally spaced covariates; otherwise, use random uniform covariates
        """
        self.function_list = function_list
        self.noise_variance = noise_variance
        self.n_datapoints = n_datapoints
        self.x_min = x_min
        self.x_max = x_max
        self.equally_spaced = equally_spaced

    def generate_data(self):
        """
        Generate synthetic tabular data with the response variable y.

        Returns:
            X: Matrix containing the training data, shape (n_datapoints x len(function_list))
            y: Response variable
            Y: Matrix of individual function responses (before summing), shape (len(function_list) x n_datapoints)
            noise: Gaussian noise added to the response variable
        """
        X = self._generate_covariates()
        Y = self._generate_response(X)
        noise = np.random.normal(0, np.sqrt(self.noise_variance), size=self.n_datapoints)
        y_mean = np.sum(Y, axis=0)
        y = y_mean + noise

        self._plot_functions(X, Y, noise)
        return X, y, Y, noise

    def _generate_covariates(self):
        """
        Generate the covariates (X) for the data.
        """
        if self.equally_spaced:
            X = np.repeat(np.linspace(self.x_min, self.x_max, self.n_datapoints).reshape(-1, 1), len(self.function_list), axis=1)
        else:
            X = np.random.uniform(self.x_min, self.x_max, size=(self.n_datapoints, len(self.function_list)))
        return X

    def _generate_response(self, X):
        """
        Generate the response variable Y based on the given functions and zero-center the effects.
        """
        num_functions = len(self.function_list)
        Y = np.zeros((num_functions, self.n_datapoints))
        for i, func in enumerate(self.function_list):
            Y[i] = func(X[:, i])
        Y = Y - np.mean(Y, axis=1).reshape(-1, 1)  # Zero-center the effects
        return Y

    def _plot_functions(self, X, Y, noise):
        """
        Plot the functions and the noisy data.
        """
        num_functions = len(self.function_list)
        if num_functions > 1:
            fig, axs = plt.subplots(1, num_functions, figsize=(15, 5))
            for i in range(num_functions):
                axs[i].scatter(X[:, i], Y[i] + noise, s=5, label=f'f_{i + 1} with noise')
                axs[i].scatter(X[:, i], Y[i], s=5, label=f'f_{i + 1}', c="red")
                axs[i].set_xlabel(f'x_{i + 1}')
                axs[i].set_ylabel('y')
                axs[i].legend()
        else:
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            axs.scatter(X[:, 0], Y[0] + noise, s=5, label=f'f_1 with noise')
            axs.scatter(X[:, 0], Y[0], s=5, label=f'f_1', c="red")
            axs.set_xlabel(f'x_1')
            axs.set_ylabel('y')
            axs.legend()

        plt.tight_layout()
        plt.show()


# Example Usage
f1 = lambda x: np.sin(2 * np.pi * x) + 3
f2 = lambda x: 2 * x
f3 = lambda x: x ** 4

DATA_FUNCTIONS = [f1, f2, f3]
generator = SyntheticDataGenerator(DATA_FUNCTIONS, noise_variance=1e-2, equally_spaced=False)
X_tabular, y_response, Y_response, y_noise = generator.generate_data()
