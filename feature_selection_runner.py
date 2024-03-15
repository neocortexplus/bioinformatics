from fs_methods import FeatureSelector
from data_manipulator import DataManipulator
from data_handler import GEODataManager
from xgboost import XGBClassifier
from threshold_applier import ThresholdApplier
import pandas as pd
from fs_methods import FeatureSelector
from data_manipulator import DataManipulator
from data_handler import GEODataManager
import pandas as pd

class FeatureSelectionRunner:
    def __init__(self, dataframe, label_column, n_runs=10, augmentation_methods=['none', 'bootstrap'], selection_methods=['xgboost', 'ttest', 'mutual_info', 'logistic_regression'], aggregation_methods=['mean', 'median', 'sum', 'all'], model_params=None):
        self.dataframe = dataframe
        self.label_column = label_column
        self.n_runs = n_runs
        self.augmentation_methods = augmentation_methods
        self.selection_methods = selection_methods
        self.aggregation_methods = aggregation_methods
        self.model_params = model_params or {
            'xgboost': {'max_depth': 3, 'n_estimators': 100},
            'random_forest': {'n_estimators': 100, 'max_depth': None},
            'anova': {'k': -1},
            'mutual_info': {'k': -1},
            'logistic_regression': {'n': -1, 'C': 1.0}
        }
        self.feature_importances = {}

    def augment_and_select_features(self):
        all_results = {}
        for augmentation_method in self.augmentation_methods:
            for selection_method in self.selection_methods:
                combined_importances = pd.DataFrame()

                for run in range(1, self.n_runs + 1):
                    print(f"Run {run}/{self.n_runs}, Augmentation: {augmentation_method}, Selection: {selection_method}")
                    augmented_data = self.dataframe

                    if augmentation_method == 'bootstrap':
                        augmented_data = DataManipulator(augmented_data).bootstrap_sample()
                    elif augmentation_method == 'none':
                        augmented_data = self.dataframe

                    selector = FeatureSelector(dataframe=augmented_data, label_column=self.label_column)
                    selector.encode_labels()
                    selector.split_data()

                    # Dynamically call the appropriate feature selection method on the FeatureSelector instance
                    method_to_call = getattr(selector, f'train_{selection_method}')
                    method_to_call(**self.model_params.get(selection_method, {}))

                    # Adjusted to accommodate the DataFrame structure for importances
                    if selector.feature_importances is not None:
                        # Ensure the importances are sorted in descending order within the FeatureSelector method implementations
                        importances_df = selector.feature_importances.rename(columns={'Importance': f'importance_run_{run}'}).set_index('Feature')
                        combined_importances = pd.concat([combined_importances, importances_df], axis=1)

                # Store combined_importances directly without redundant list conversion
                if not combined_importances.empty:
                    all_results[f"{augmentation_method}_{selection_method}"] = combined_importances

        self.feature_importances = all_results

    def apply_aggregation_methods(self):
        """
        If 'all' is specified in aggregation_methods, applies mean, median, and sum 
        aggregations to the feature importances and adds the results as new columns 
        to the respective DataFrames. It can also handle other specified aggregation methods.
        """
        # Define the default aggregation operations
        default_aggregations = ['mean', 'median', 'sum']
        
        # Determine the aggregation operations to apply based on the presence of 'all'
        if 'all' in self.aggregation_methods:
            aggregation_methods_to_apply = default_aggregations
        else:
            # This allows for future customization or extension of aggregation methods
            aggregation_methods_to_apply = [method for method in self.aggregation_methods if method in default_aggregations]
        
        # Apply the determined aggregation methods
        for key, df in self.feature_importances.items():
            for method in aggregation_methods_to_apply:
                agg_result = getattr(df, method)(axis=1)
                df[method] = agg_result  # Add as a new column to the DataFrame
    def save_feature_importances_to_csv(self):
        """
        Saves each DataFrame in self.feature_importances to a CSV file,
        including the feature names as index and the aggregation columns.
        """
        for key, df in self.feature_importances.items():
            # Construct a filename based on the key
            filename = f"{key}_feature_importances.csv"
            # Save the DataFrame to CSV
            df.to_csv(filename, header=True, index=True)
            print(f"Saved {filename}")



# Example usage
if __name__ == "__main__":
    filename = "final_data.csv"
    df = GEODataManager.load_csv(filename)
    runner = FeatureSelectionRunner(
        dataframe=df,
        label_column='Label',
        n_runs=2,
        augmentation_methods=['none', 'bootstrap'],
        selection_methods=['xgboost', 'ttest', 'logistic_regression'],
        aggregation_methods=['mean', 'median', 'sum'],
        model_params={
            'xgboost': {'max_depth': 4, 'n_estimators': 100},
            'anova': {'k': -1},
            'mutual_info': {'k': -1},
            'logistic_regression': {'n': -1, 'C': 0.5}
        }
    )

    
    runner.augment_and_select_features()
    runner.apply_aggregation_methods()
    runner.save_feature_importances_to_csv()

    thresholds = [1, 'KDE', 'GMM']  # Apply 1% threshold, KDE, and GMM thresholds
    methods_to_apply = 'all'  
    selection_methods_to_apply = 'all'  

    threshold_applier = ThresholdApplier(thresholds, methods_to_apply, selection_methods_to_apply)
    threshold_applier.apply_thresholds(runner.feature_importances)