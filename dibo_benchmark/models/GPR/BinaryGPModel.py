import pibob
# from pibob.models.GPR.GP_base import ExactGPModel  # <- not used for classification
import gpytorch
import torch
from pibob.models.GPR.model import model_base
from typing_extensions import Union
import numpy as np

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPClassificationCore(gpytorch.models.ApproximateGP):
    """
    Variational GP model for binary classification.
    """

    def __init__(self, inducing_points: torch.Tensor, input_dim: int):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)





class BinaryGPModel(model_base):
    """
    BinaryGPModel for **binary classification** using a variational GP
    (ApproximateGP + BernoulliLikelihood).

    y_data must be 0/1 (float or int).
    """

    def __init__(self, x_data, y_data, x_err=None, y_err=None, n=None, **kwargs):
        """
        Args:
            x_data: array-like (N, D)
            y_data: array-like (N,) with labels in {0, 1}
            x_err, y_err: kept for API compatibility (not used in likelihood)
            n: number of input features
            kwargs:
                n_inducing (int): number of inducing points (default: min(64, N))
        """
        # Convert to tensors if numpy
        if isinstance(x_data, np.ndarray):
            if n is None:
                n = x_data.shape[1]
            self.x_data = torch.from_numpy(x_data)
            self.y_data = torch.from_numpy(y_data)
            self.x_err = torch.from_numpy(x_err) if x_err is not None else None
            self.y_err = torch.from_numpy(y_err) if y_err is not None else None
        else:
            if n is None:
                n = x_data.size(1)
            self.x_data = x_data
            self.y_data = y_data
            self.x_err = x_err
            self.y_err = y_err

        self.n = n

        # Dtypes
        self.x_data = self.x_data.to(torch.get_default_dtype())
        # labels must be float 0/1 for BernoulliLikelihood
        self.y_data = self.y_data.to(torch.get_default_dtype())
        if self.x_err is not None:
            self.x_err = self.x_err.to(torch.get_default_dtype())
        if self.y_err is not None:
            self.y_err = self.y_err.to(torch.get_default_dtype())

        # Hyperparameters from kwargs
        self.n_inducing = kwargs.get("n_inducing", None)

        self.data_statistics()
        self.standadize_x()
        self.standadize_y()  # will just copy labels

    def data_statistics(self):
        """
        Computes mean and std for the input data.

        For classification, we **do not** standardize y.
        """
        self.m_x = torch.mean(self.x_data, 0)
        self.s_x = torch.std(self.x_data, 0)

        # Avoid zero std to prevent division by zero
        self.s_x = torch.where(self.s_x == 0, torch.ones_like(self.s_x), self.s_x)

    def standadize_x(self):
        """
        Standardizes the input data and its error.
        """
        self.x_data_s = (self.x_data - self.m_x) / self.s_x
        if self.x_err is not None:
            self.x_err_s = (self.x_err) / self.s_x
        else:
            self.x_err_s = None

    def standadize_y(self):
        """
        For classification, we **do not** standardize y.
        Just ensure it's a float tensor with values in {0,1}.
        """
        # y_data_s is just labels in {0,1}
        self.y_data_s = self.y_data

    def destandadize(self):
        """
        For classification, predictions are probabilities in [0,1].
        We keep them as-is (no destandardization).
        """
        self.y_pred = self.y_pred_s
        self.y_pred_err = self.y_pred_err_s

    def define(self):
        """
        Defines the variational GP classification model and the Bernoulli likelihood.
        """
        N = self.x_data_s.size(0)

        if self.n_inducing is None:
            self.n_inducing = min(64, N)

        # Simple choice: random subset of training points as inducing
        inducing_indices = torch.randperm(N)[: self.n_inducing]
        inducing_points = self.x_data_s[inducing_indices]

        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.model = GPClassificationCore(inducing_points, self.n)

    def train(self, train_iter=2000, lr=0.001):
        """
        Trains the GP classifier using variational ELBO.

        Args:
            train_iter (int): Number of training iterations.
            lr (float): Learning rate.
        """
        self.model.train()
        self.likelihood.train()

        print("Learning rate: ", lr)
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=lr,
        )

        mll = gpytorch.mlls.VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=self.y_data_s.size(0),
        )

        for i in range(train_iter):
            optimizer.zero_grad()
            output = self.model(self.x_data_s)
            loss = -mll(output, self.y_data_s)
            loss.backward()

            if np.mod(i, max(1, train_iter // 10)) == 0:
                formatted_iter = "Iter %d/%d - Loss: %.4f" % (
                    i + 1,
                    train_iter,
                    loss.item(),
                )
                lengthscales = (
                    self.model.covar_module.base_kernel.lengthscale[0]
                    .detach()
                    .cpu()
                    .tolist()
                )
                formatted_lengthscales = " ".join(["%.3f" % num for num in lengthscales])
                print(formatted_iter + "   lengthscales: " + formatted_lengthscales)

            optimizer.step()

    def predict(self, x_pred):
        """
        Predicts class probabilities for new inputs.

        Args:
            x_pred (torch.Tensor or np.ndarray): Input data for prediction, shape (N*, D)

        Returns:
            y_pred: P(y=1 | x_pred)
            y_pred_err: sqrt(var) of predictive Bernoulli (approx.)
        """
        if isinstance(x_pred, np.ndarray):
            x_pred = torch.from_numpy(x_pred)

        x_pred = x_pred.to(torch.get_default_dtype())

        self.x_pred = (x_pred - self.m_x) / self.s_x
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_f = self.model(self.x_pred)
            observed_pred = self.likelihood(latent_f)

            # For BernoulliLikelihood: mean in [0,1] is P(y=1)
            self.y_pred_s = observed_pred.mean
            # Variance of Bernoulli predictions
            var = observed_pred.variance
            self.y_pred_err_s = torch.sqrt(var)

        self.destandadize()
        return self.y_pred, self.y_pred_err

    def save_GP(self, path):
        """
        Saves the trained GP classifier and related data.
        """
        torch.save(self.model.state_dict(), path + "model.pth")
        torch.save(self.likelihood.state_dict(), path + "likelihood.pth")
        torch.save(self.x_data_s, path + "x_data.pt")
        if self.x_err_s is not None:
            torch.save(self.x_err_s, path + "x_err_s.pt")
        torch.save(self.y_data_s, path + "y_data.pt")
        torch.save(self.m_x, path + "m_x.pt")
        torch.save(self.s_x, path + "s_x.pt")
        torch.save(self.n, path + "n.pt")
        torch.save(self.n_inducing, path + "n_inducing.pt")

    def Load_GP(self, path):
        """
        Loads a saved GP classifier and related data.
        """
        model_state_dict = torch.load(path + "model.pth")
        likelihood_state_dict = torch.load(path + "likelihood.pth")
        self.x_data_s = torch.load(path + "x_data.pt")

        try:
            self.x_err_s = torch.load(path + "x_err_s.pt")
        except FileNotFoundError:
            self.x_err_s = None

        self.y_data_s = torch.load(path + "y_data.pt")
        self.m_x = torch.load(path + "m_x.pt")
        self.s_x = torch.load(path + "s_x.pt")
        self.n = torch.load(path + "n.pt")
        self.n_inducing = torch.load(path + "n_inducing.pt")

        # Rebuild model & likelihood
        self.define()
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        self.likelihood.load_state_dict(likelihood_state_dict)
        self.likelihood.eval()
