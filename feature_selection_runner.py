import pandas as pd
from data_Manipulator import DataAugmentation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from data_handler import GEODataManager  # Ensure this is implemented correctly

class FeatureSelectionRunner:
    def __init__(self, dataframe, label_column, n_runs=10, augmentation_method='bootstrap', feature_selection_model=RandomForestClassifier()):
        self.dataframe = dataframe
        self.label_column = label_column
        self.n_runs = n_runs
        self.augmentation_method = augmentation_method
        self.feature_selection_model = feature_selection_model
        self.results = []

    def augment_and_select_features(self):
        for run in range(self.n_runs):
            if self.augmentation_method == 'bootstrap':
                augmented_data = DataAugmentation(self.dataframe).bootstrap_sample(1)[0]
            else:
                # Add other augmentation methods as needed
                augmented_data = self.dataframe

            # Splitting augmented data into features and labels
            X = augmented_data.drop(self.label_column, axis=1)
            y = augmented_data[self.label_column]
            
            # Fitting the feature selection model
            self.feature_selection_model.fit(X, y)
            selector = SelectFromModel(self.feature_selection_model, prefit=True)
            selected_features = X.columns[(selector.get_support())]

            # Storing results
            self.results.append({
                'run': run + 1,
                'selected_features': list(selected_features)
            })

    def save_results_to_csv(self, filename="feature_selection_results.csv"):
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


# Example usage
if __name__ == "__main__":
    filename = "final_data.csv"
    df = GEODataManager.load_csv(filename)  # Adjust this to your actual implementation
    runner = FeatureSelectionRunner(dataframe=df, label_column='Label', n_runs=10, augmentation_method='bootstrap')
    runner.augment_and_select_features()
    runner.save_results_to_csv()
