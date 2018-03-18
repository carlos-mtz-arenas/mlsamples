import pandas as pd
from sklearn.model_selection import train_test_split


class AbstractRegressor:
    """
    Machine learning processor template for regressors so that
    we just have to override a few methods to specify
    the implementation that we require.
    """

    def __init__(self, dataset, y_column, unnecessary_columns=None, plot_results=False):
        self.dataset = dataset
        self.y_column = y_column
        self.unnecessary_columns = unnecessary_columns
        self.y = None
        self.plot_results = plot_results
        self.original_dataframe = None

    def fit(self):
        x = self.read_file()
        x = self.preprocess_data_if_needed(x)
        x_train, x_test, y_train, y_test = self.split_data(x)
        trained_model = self.fit_model(x_train, y_train)
        self.test_model(trained_model, x_train, y_train, x_test, y_test)
        if self.plot_results:
          self.plot_stuff(trained_model, x_train, y_train, x_test, y_test)
        return trained_model

    def plot_stuff(self, trained_model, x_train, y_train, x_test, y_test):
      raise NotImplementedError('There\'s no implementation for plotting the results')


    def fit_model(self, x_train, y_train):
        """
        Trains the model used the given regressor on concrete class.
        :param x_train: X matrix for training.
        :param y_train: Y feature vector for training.
        :return: The trained model.
        """
        raise NotImplementedError('Fit model should be implemented by child class')

    def test_model(self, model, x, y, x_test, y_test):
        """
        Tests the trained model, uses the training and testing data.
        :param model: The trained model.
        :param x: X data used on training.
        :param y: Y data used on training.
        :param x_test: X data for testing (not used on training).
        :param y_test: Y data for testing (not used on training).
        """
        raise NotImplementedError('Test trained data should be implemented by child class')

    def read_file(self):
        """
        Reads the file for the dataset using pandas.
        :return: the dataset matrix.
        """
        data = pd.read_csv(self.dataset)
        self.original_dataframe = data.copy()

        self.y = data[[self.y_column]]

        # remove the dependant variable column
        data = data.drop(columns=([self.y_column] + self.unnecessary_columns))
        return data

    def preprocess_data_if_needed(self, x):
        print('Using default implementation, returning just X')
        return x

    def split_data(self, x):
        """
        Splits the data into training and testing data in a 70-30 ratio.
        :param x: The X data to be split.
        :return: The split data to train and test.
        """
        return train_test_split(x, self.y, test_size=0.3, random_state=0)
