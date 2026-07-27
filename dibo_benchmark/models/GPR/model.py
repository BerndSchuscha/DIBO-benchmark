# Base class for all models


from abc import ABC, abstractmethod


class model_base(ABC):
    """
    Abstract base class for all models.

    This class enforces the implementation of the `train` and `predict` methods
    in any subclass.

    Methods:
        train: Abstract method to train the model.
        predict: Abstract method to make predictions with the model.
    """

    @abstractmethod
    def __init__(self, x_data, y_data, x_err, y_err):
        """
        Initializes the model base class.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Trains the model.
        """
        pass

    @abstractmethod
    def predict(self, x_pred):
        """
        Makes predictions with the model.
        """
        pass

    @abstractmethod
    def define(self):
        """
        Defines the model.
        """
        pass

    def standadize_x(self):
        """
        Standardizes the input data and its error.
        """
        pass

    def standadize_y(self):
        """
        Standardizes the output data and its error.
        """
        pass

    def destandadize(self):
        """
        Converts standardized predictions back to the original scale.
        """
        pass
