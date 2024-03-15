import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import numpy as np

class KDEThreshold:
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def calculate_threshold(self, data):
        """
        Calculates a threshold based on the density of data points.
        The method looks for a significant drop in the density as the cutoff for the number of features to keep.
        """
        kde = gaussian_kde(data, bw_method=self.bandwidth)
        support = np.linspace(data.min(), data.max(), len(data))
        density = kde(support)
        return self.find_significant_drop(density, support)

    def find_significant_drop(self, density, support):
        """
        Identifies the first significant drop in density to determine the cutoff threshold.
        This method is a simplified heuristic and might need adjustments for different datasets.
        """
        drop_threshold = np.diff(density).max() * -0.05  # Defines what we consider a significant drop
        for i in range(1, len(density)):
            if np.diff(density)[i-1] < drop_threshold:
                return support[i]
        return support[-1]

class GMMThreshold:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def calculate_threshold(self, data):
        """
        Calculates a threshold based on Gaussian Mixture Model fitting.
        The method selects the mean of the least important component as the threshold.
        """
        gmm = GaussianMixture(n_components=self.n_components, random_state=0)
        gmm.fit(data.reshape(-1, 1))
        means = gmm.means_.flatten()
        return np.min(means)
        
class ThresholdApplier:
    def __init__(self, thresholds, methods='all', selection_methods='all'):
        """
        Initializes the ThresholdApplier with user-defined thresholds, methods, and selection_methods.
        :param thresholds: A list of thresholds (in percentages), 'KDE', 'GMM', or a combination of these.
        :param methods: A list of methods to which thresholds should be applied, or 'all' for all methods.
        :param selection_methods: Feature selection methods to apply thresholds to, or 'all' for all methods.
        """
        self.thresholds = thresholds
        self.methods = methods
        self.selection_methods = selection_methods

    def apply_thresholds(self, feature_importances):
        for key, df in feature_importances.items():
            selection_method = key.split('_')[1]
            applicable_methods = self.methods if self.methods != 'all' else df.columns.tolist()

            if self.selection_methods == 'all' or selection_method in self.selection_methods:
                for method in applicable_methods:
                    if method in df.columns:
                        original_values = df[method].copy()  # Copy the original column values
                        for threshold in self.thresholds:
                            if isinstance(threshold, str) and threshold.upper() in ['KDE', 'GMM']:
                                self.apply_complex_threshold(df, method, threshold, original_values)
                            elif isinstance(threshold, (int, float)):  # Handle percentage thresholds
                                self.apply_percentage_threshold(df, method, threshold, original_values)

    def apply_percentage_threshold(self, df, method, threshold, original_values):
        n_elements_to_retain = int(len(df) * (threshold / 100.0))
        if n_elements_to_retain > 0:
            df_sorted = original_values.sort_values(ascending=False)
            top_n_indices = df_sorted.head(n_elements_to_retain).index
            df[method] = 0  # Reset the column to 0
            df.loc[top_n_indices, method] = original_values.loc[top_n_indices]
        else:
            df[method] = 0

    def apply_complex_threshold(self, df, method, threshold_type, original_values):

        non_zero_data = original_values[original_values > 0].values
        
        if threshold_type.upper() == 'KDE':
            threshold_calculator = KDEThreshold()
            num_features_to_keep = threshold_calculator.calculate_threshold(non_zero_data)
        elif threshold_type.upper() == 'GMM':
            threshold_calculator = GMMThreshold()
            num_features_to_keep = threshold_calculator.calculate_threshold(non_zero_data)
        else:
            raise ValueError("Unsupported threshold type: {}".format(threshold_type))
        
        # Sort the original values in descending order to keep the top N features
        sorted_indices = np.argsort(-non_zero_data)  # Note the negative sign for descending sort
        # Create a mask to retain only the top N features, setting others to 0
        filtered_data = non_zero_data[non_zero_data >= num_features_to_keep]
        # Apply the mask to retain top N features and set others to 0
        df[method] = df[df[method] >= num_features_to_keep][method]