# Assuming FeatureSelector is saved in 'feature_selector.py'
from fs_methods import FeatureSelector
from data_manipulator import DataManipulator  # Make sure this is the correct import path
from data_handler import GEODataManager  # Ensure this is correctly implemented
import pandas as pd

class FeatureSelectionRunner:
    def __init__(self, dataframe, label_column, n_runs=10, augmentation_method='bootstrap', selection_method='xgboost'):
        self.dataframe = dataframe
        self.label_column = label_column
        self.n_runs = n_runs
        self.augmentation_method = augmentation_method
        self.selection_method = selection_method
        self.feature_importances = pd.DataFrame()

    def augment_and_select_features(self):
        all_importances = []
        
        for run in range(self.n_runs):
            print(f"Run {run+1}/{self.n_runs}")
            if self.augmentation_method == 'bootstrap':
                # Ensure augmented_data is a DataFrame, not a list
                augmented_data = DataManipulator(self.dataframe).bootstrap_sample(1)
            else:
                augmented_data = self.dataframe

            # Initialize FeatureSelector with the current augmented dataset
            selector = FeatureSelector(dataframe=augmented_data, label_column=self.label_column)
            selector.encode_labels()
            selector.split_data()

            if self.selection_method == 'xgboost':
                selector.train_xgboost()
            elif self.selection_method == 'random_forest':
                selector.train_random_forest()
            else:
                print("Invalid selection method specified.")
                return
            
            # Append the feature importances for this run to all_importances
            if selector.feature_importances is not None:
                all_importances.extend(selector.feature_importances.values.flatten().tolist())
        
        # Averaging feature importances across all runs, if any runs were completed
        if all_importances:
            avg_importances = pd.Series(all_importances).groupby(selector.feature_importances['Feature']).mean().reset_index(name='Importance')
            self.feature_importances = avg_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    def save_results_to_csv(self, filename="feature_selection_results.csv"):
        if not self.feature_importances.empty:
            self.feature_importances.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        else:
            print("No feature importances to save.")

# Example usage
if __name__ == "__main__":
    filename = "final_data.csv"
    df = GEODataManager.load_csv(filename)  # Ensure this method correctly loads your dataframe
    runner = FeatureSelectionRunner(dataframe=df, label_column='Label', n_runs=10, augmentation_method='bootstrap', selection_method='xgboost')
    runner.augment_and_select_features()
    runner.save_results_to_csv()
