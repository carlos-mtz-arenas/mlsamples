from templates.abstractregressor import AbstractRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegressor(AbstractRegressor):

    def __init__(self):
        super(LinearRegressor, self)\
            .__init__('training_data/vgsales.csv',
            'Global_Sales',
            ['Name', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Rank'])

    def fit_model(self, x_train, y_train):
        print('Training model')
        model = LinearRegression()
        model.fit(x_train, y_train)
        return model

    def test_model(self, model, x, y, x_test, y_test):
        training_mean_error = mean_absolute_error(y, model.predict(x))
        print('TRAINING mean absolute error %.4f' % training_mean_error)
        test_mean_error = mean_absolute_error(y_test, model.predict(x_test))
        print('TEST mean absolute error %.4f' % test_mean_error)

    def preprocess_data_if_needed(self, x):
      # apply one hot encoding to the categorical features
      categorical_features = ['Platform', 'Genre', 'Publisher']
      x = pd.get_dummies(x, columns=categorical_features)
      return x


if __name__ == '__main__':
    regressor = LinearRegressor()
    regressor.fit()

