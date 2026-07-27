import pibob
import gpytorch
from typing import Callable, Dict, Iterable, Optional, Tuple, Union


class ExactGPModel(gpytorch.models.ExactGP):

    """
    A Gaussian Process model that uses exact inference.

    This model is initialized with training data and a likelihood function, and uses a constant mean function
    and a scaled Radial Basis Function (RBF) kernel with Automatic Relevance Determination (ARD).

    Attributes:
        mean_module (gpytorch.means.ConstantMean): The mean module for the GP, which returns a constant mean.
        covar_module (gpytorch.kernels.ScaleKernel): The covariance module, which uses a scaled RBF kernel.
    """

    def __init__(self, train_x, train_y, likelihood, n):
        """
        Initializes the ExactGPModel with training data and a likelihood function.

        Args:
            train_x (torch.Tensor): Training input data.
            train_y (torch.Tensor): Training target data.
            likelihood (gpytorch.likelihoods.Likelihood): The likelihood function.
            n (int): Number of input dimensions for the RBF kernel with ARD.
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(
            nu=0.5,
            ard_num_dims=n,
            lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(0.01, 100, sigma=0.01)))

    def forward(self, x):
        """
        Defines the forward pass of the GP model.

        This method computes the mean and covariance of the Gaussian process at the input points x.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            gpytorch.distributions.MultivariateNormal: The multivariate normal distribution representing the GP's predictions.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # type: ignore
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

