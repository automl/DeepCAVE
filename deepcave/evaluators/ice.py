import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor


class ICE:
    def __init__(self, data={}):
        self.model = None

        # Make sure to have int keys
        self.data = {}

        for k, (X, Y) in data.items():

            X = np.array(X)
            Y = np.array(Y)

            self.data[int(k)] = (X, Y)

    def get_data(self):
        return self.data

    def fit(self, configspace, X, Y, seed=0):
        """
        Args:
            s (int): The id of the requested hyperparameter.
        """

        # Train random forest here
        if self.model is None:
            self.model = RandomForestRegressor(random_state=seed)
            self.model.fit(X, Y)

        for hp_name in configspace.get_hyperparameter_names():
            s = configspace.get_idx_by_hyperparameter_name(hp_name)

            shape = X.shape
            X_ice = np.zeros((shape[0], *shape))
            y_ice = np.zeros((shape[0], shape[0]))

            # Iterate over data points
            for i, _ in enumerate(X):
                X_copy = X.copy()

                # Intervention
                # Take the value of i-th data point and set it to all others
                # We basically fix the value
                X_copy[:, s] = X_copy[i, s]
                X_ice[i] = X_copy

                # Then we do a prediction with the new data
                y_ice[i] = self.model.predict(X_copy)

            self.data[int(s)] = (X_ice, y_ice)

    def get_ice_data(self, s, centered=False):
        if s not in self.data:
            return [], []

        (X_ice, y_ice) = self.data[s]

        all_x = []
        all_y = []

        for i in range(X_ice.shape[0]):
            x = X_ice[:, i, s]
            y = y_ice[:, i]

            # We have to sort x because they might be not
            # in the right order
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]

            # Or all zero centered (c-ICE)
            if centered:
                y = y - y[0]

            all_x.append(x)
            all_y.append(y)

        return all_x, all_y

    def get_pdp_data(self, s):
        if s not in self.data:
            return [], []

        (X_ice, y_ice) = self.data[s]

        n = y_ice.shape[0]

        # Take all x_s instance value
        x = X_ice[:, 0, s]

        y = []
        # Simply take all values and mean them
        for i in range(n):
            m = np.mean(y_ice[i, :])
            y.append(m)

        y = np.array(y)

        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        # Let's calculate uncertainties here
        m_s = np.mean(y)

        return x, y
