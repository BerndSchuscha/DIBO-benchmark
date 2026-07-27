import pibob
from pibob.models.GPR.GP_base import ExactGPModel
import gpytorch
import torch
from pibob.models.GPR.model import model_base
from typing_extensions import Union
import numpy as np

# pick one dtype (I suggest float64 for GP stability)
dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPModel(model_base):
    """
    HGPModel is a Gaussian Process Model class that handles training,
    standardization, and prediction for a given set of data.

    Attributes:
        x_data (torch.Tensor): Input data.
        y_data (torch.Tensor): Output data.
        x_err (torch.Tensor): Error in input data.
        y_err (torch.Tensor): Error in output data.
        n (int): Number of input features.
        m_x (torch.Tensor): Mean of x_data.
        s_x (torch.Tensor): Standard deviation of x_data.
        m_y (torch.Tensor): Mean of y_data.
        s_y (torch.Tensor): Standard deviation of y_data.
        x_data_s (torch.Tensor): Standardized input data.
        x_err_s (torch.Tensor): Standardized input error.
        y_data_s (torch.Tensor): Standardized output data.
        y_err_s (torch.Tensor): Standardized output error.
        y_pred (torch.Tensor): Predicted output data.
        y_pred_err (torch.Tensor): Error in predicted output data.
        likelihood (gpytorch.likelihoods.FixedNoiseGaussianLikelihood): Likelihood function for GP model.
        model (ExactGPModel): Gaussian Process model.
    """

    def __init__(self, x_data, y_data, x_err, y_err, n=None, **kwargs):
        """
        Initializes the HGPModel with input data, output data, and their respective errors.

        Args:
            x_data (torch.Tensor): Input data.
            y_data (torch.Tensor): Output data.
            x_err (torch.Tensor): Error in input data.
            y_err (torch.Tensor): Error in output data.
            n (int, optional): Number of input features. Defaults to None.
        """
        if isinstance(x_data, np.ndarray):
            if n == None:
                n = np.shape(x_data)[1]
            self.x_data = torch.from_numpy(x_data)
            self.n = n
            self.y_data = torch.from_numpy(y_data)
            self.x_err = torch.from_numpy(x_err)
            self.y_err = torch.from_numpy(y_err)
        else:
            if n == None:
                n = x_data.size(1)
            self.x_data = x_data
            self.n = n
            self.y_data = y_data
            self.x_err = x_err
            self.y_err = y_err

        self.x_data = self.x_data.to(torch.get_default_dtype())
        self.y_data = self.y_data.to(torch.get_default_dtype())
        self.x_err = self.x_err.to(torch.get_default_dtype())
        self.y_err = self.y_err.to(torch.get_default_dtype())

        self.data_statistics()
        self.standadize_x()
        self.standadize_y()

    def data_statistics(self):
        """
        Computes the mean and standard deviation for the input and output data.
        """
        self.m_x = torch.mean(self.x_data, 0)
        self.s_x = torch.std(self.x_data, 0)
        self.m_y = torch.mean(self.y_data)
        self.s_y = torch.std(self.y_data)

    def standadize_x(self):
        """
        Standardizes the input data and its error.
        """
        self.x_data_s = (self.x_data - self.m_x) / self.s_x
        self.x_err_s = (self.x_err) / self.s_x

    def standadize_y(self):
        """
        Standardizes the output data and its error.
        """
        self.y_data_s = (self.y_data - self.m_y) / self.s_y
        self.y_err_s = (self.y_err) / self.s_y

    def destandadize(self):
        """
        Converts standardized predictions back to the original scale.
        """
        self.y_pred = self.y_pred_s * self.s_y + self.m_y
        self.y_pred_err = self.y_pred_err_s * self.s_y

    def define(self):
        """
        Defines the GP model and the likelihood function.
        """
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(self.x_data_s, self.y_data_s, self.likelihood, self.n)

    def train(self, train_iter=2000, lr=0.001):
        """
        Trains the GP model.

        Args:
            train_iter (int): Number of training iterations. Defaults to 200000.
            lr (float): Learning rate. Defaults to 0.1.
        """
        self.model.train()
        self.likelihood.train()
        # Includes GaussianLikelihood parameters
        print("Learning rate: ", lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(train_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.x_data_s)
            # Calc loss and backprop gradients

            loss = -mll(output, self.y_data_s)  # type: ignore
            loss.backward()
            if np.mod(i, train_iter / 10) == 0 and True:

                formatted_iter = "Iter %d/%d - Loss: %.4f" % (
                    i + 1,
                    train_iter,
                    loss.item(),
                )
                formatted_lengthscales = " ".join(
                    [
                        "%.3f" % num
                        for num in (
                            self.model.covar_module.base_kernel.lengthscale[0]
                        ).tolist()
                    ]
                )  # type: ignore

                print(formatted_iter + "   lengthscales: " + formatted_lengthscales)
            optimizer.step()

    def predict(self, x_pred):
        """
        Makes predictions using the trained GP model.

        Args:
            x_pred (torch.Tensor): Input data for predictions.
        """
        x_pred = x_pred.to(torch.get_default_dtype())

        self.x_pred = (x_pred - self.m_x) / self.s_x
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(self.x_pred))
            lower, _ = observed_pred.confidence_region()  # type: ignore
            self.y_pred_s = observed_pred.mean
            self.y_pred_err_s = (self.y_pred_s - lower) / 2

        self.destandadize()
        return self.y_pred, self.y_pred_err

    def save_GP(self, path):
        """
        Saves the trained GP model and related data to files.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), path + "model.pth")
        torch.save(self.likelihood.state_dict(), path + "likelyhood.pth")
        torch.save(self.x_data_s, path + "x_data.pt")
        torch.save(self.x_err_s, path + "x_err_s.pt")
        torch.save(self.y_err_s, path + "y_err.pt")
        torch.save(self.y_data_s, path + "y_data.pt")
        torch.save(self.m_x, path + "m_x.pt")
        torch.save(self.m_y, path + "m_y.pt")
        torch.save(self.s_x, path + "s_x.pt")
        torch.save(self.s_y, path + "s_y.pt")

    def Load_GP(self, path):
        """
        Loads a saved GP model and related data from files.

        Args:
            path (str): Path to load the model from.
        """
        model_state_dict = torch.load(path + "model.pth")
        likelihood_state_dict = torch.load(path + "likelyhood.pth")
        self.x_data_s = torch.load(path + "x_data.pt")
        self.x_err_s = torch.load(path + "x_err_s.pt")
        self.y_data_s = torch.load(path + "y_data.pt")
        self.y_err_s = torch.load(path + "y_err.pt")
        self.m_x = torch.load(path + "m_x.pt")
        self.m_y = torch.load(path + "m_y.pt")
        self.s_x = torch.load(path + "s_x.pt")
        self.s_y = torch.load(path + "s_y.pt")
        self.n = self.x_data_s.size()[1]

        self.define()

        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        self.likelihood.load_state_dict(likelihood_state_dict)
        self.likelihood.eval()
