import pandas as pd
import numpy as np
from sklearn.utils import resample
from data_handler import GEODataManager  # Ensure this is correctly implemented

class DataManipulator:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def permute_dataset(self):
        """Permute the rows of the dataset randomly."""
        return self.dataframe.sample(frac=1).reset_index(drop=True)

    def add_noise_to_data(self, n_datasets=5, noise_level=0.01):
            """
            Generate multiple noisy datasets.
            :param n_datasets: Number of noisy datasets to generate.
            :param noise_level: Standard deviation of Gaussian noise relative to the data standard deviation.
            :return: A list of DataFrames, each a noisy version of the original dataset.
            """
            noisy_datasets_list = []  # Initialize an empty list to hold the noisy datasets
            for _ in range(n_datasets):
                noisy_df = self.dataframe.copy()
                for column in noisy_df.columns:
                    if noisy_df[column].dtype in ['float64', 'float32', 'int64', 'int32']:
                        std = noisy_df[column].std()
                        noise = np.random.normal(0, std * noise_level, size=noisy_df[column].shape)
                        noisy_df[column] += noise
                noisy_datasets_list.append(noisy_df)  # Append the generated noisy dataset to the list
            return noisy_datasets_list

    def bootstrap_sample(self, n_samples=100):
        """
        Generate bootstrap samples of the dataset.
        :param n_samples: Number of bootstrap samples to generate.
        :return: A list of DataFrames, each a bootstrap sample of the original dataset.
        """
        bootstrap_samples_list = []  # Initialize an empty list to hold the bootstrap samples
        for i in range(n_samples):
            sample = resample(self.dataframe, replace=True, n_samples=len(self.dataframe), random_state=i)
            bootstrap_samples_list.append(sample)  # Append the generated sample to the list                 
        return bootstrap_samples_list

# Example usage
if __name__ == "__main__":

    filename = "final_data.csv"
    df = GEODataManager.load_csv(filename)  # Ensure this function is correctly implemented

    manipulator = DataManipulator(dataframe=df)
    
    bootstrap_samples = manipulator.bootstrap_sample(n_samples=5)
    print("Bootstrap Sample 1:", bootstrap_samples[0].head(), sep="\n")

    permuted_df = manipulator.permute_dataset()
    print("Permuted DataFrame:", permuted_df.head(), sep="\n")

    noisy_df = manipulator.add_noise_to_data()
    print("DataFrame with Noise:", noisy_df.head(), sep="\n")

