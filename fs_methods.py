import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from data_handler import GEODataManager  # Make sure this import matches your setup
from skrebate import ReliefF
from scipy.stats import mannwhitneyu
from mrmr import mrmr_classif
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from wxmethod.src.wx_hyperparam import WxHyperParameter
from wxmethod.src.wx_core import wx_slp, wx_mlp, connection_weight
import geode
from geode import chdir
from skfeature.function.similarity_based import fisher_score
from keras.utils import to_categorical


class DataFrameNormalizer:
    def __init__(self, dataframe):
        """
        Initialize the DataFrameNormalizer with a dataframe.
        """
        self.dataframe = dataframe
    
    def min_max_normalize(self, label_column):
        """
        Apply Min-Max normalization to all columns except the label column.
        """
        # Copy the dataframe to avoid modifying the original
        df_normalized = self.dataframe.copy()
        
        # Perform Min-Max normalization on all columns except the label column
        for column in df_normalized.columns:
            if column != label_column:
                df_normalized[column] = (df_normalized[column] - df_normalized[column].min()) / (df_normalized[column].max() - df_normalized[column].min())
        
        self.normalized_df = df_normalized
        return df_normalized

    def verify_dataset(self, label_column="Label"):
        """
        Provides comprehensive statistics for the normalized dataset, excluding the label column.
        """
        if self.normalized_df is None:
            print("Dataset has not been normalized yet.")
            return

        df_without_label = self.normalized_df.drop(columns=[label_column])

        overall_min = df_without_label.min().min()
        overall_max = df_without_label.max().max()
        mean = df_without_label.mean().mean()
        median = df_without_label.median().median()
        std_dev = df_without_label.std().mean()

        print(f"Overall Min Value: {overall_min}")
        print(f"Overall Max Value: {overall_max}")
        print(f"Mean Value: {mean}")
        print(f"Median Value: {median}")
        print(f"Standard Deviation: {std_dev}")

class StabilityCalculator:
    def __init__(self, top_features_df):
        """
        Initializes the StabilityCalculator with a DataFrame of top features.
        
        Parameters:
        - top_features_df: DataFrame, where each column represents the top features selected in one iteration.
        """
        self.top_features_df = top_features_df

    def calculate_kenchev_stability_index(self):
        """
        Calculates the Kenchev Stability Index for feature selection results provided in a DataFrame.
        
        Returns:
        - Kenchev Stability Index: A float value representing the stability of the feature selection process.
        """
        # Number of iterations (columns in the DataFrame)
        N = self.top_features_df.shape[1]

        # Number of top features selected in each iteration (rows in the DataFrame)
        k = self.top_features_df.shape[0]

        # Flatten the DataFrame to a list of lists (each sublist represents an iteration)
        features_all_iterations = [self.top_features_df[column].tolist() for column in self.top_features_df.columns]

        # Calculate pairwise overlaps and sum them up
        sum_overlaps = 0
        for i in range(N):
            for j in range(i + 1, N):
                sum_overlaps += len(set(features_all_iterations[i]).intersection(set(features_all_iterations[j])))

        # Compute Kenchev Stability Index
        K = sum_overlaps / (N * (N - 1) / 2 * k)

        return K
    
class FeatureSelector:
    def __init__(self, dataframe, label_column):
        self.dataframe = dataframe
        self.label_column = label_column
        self.model = None
        self.feature_importances = None
        self.update_X_y()

    def update_X_y(self):
        """Updates the X and y attributes based on the current state of the dataframe."""
        self.X = self.dataframe.drop(self.label_column, axis=1)
        self.y = self.dataframe[self.label_column]


    
    def encode_labels(self):
        le = LabelEncoder()
        self.y_encoded = le.fit_transform(self.y)

    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=test_size)

    def train_xgboost(self, **kwargs):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **kwargs)
        self.model.fit(self.X_train, self.y_train)
        self.feature_importances = pd.DataFrame({'Feature': self.X.columns, 'Importance': self.model.feature_importances_}).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    def train_chdir(self, num_features=20):

        # Convert X to a transposed numpy array for CHDIR processing

        data_np = self.X.T.to_numpy()
        sampleclass = [1 if sample == 'normal' else 2 for sample in self.y]

        # Call the CHDIR function, assuming it is available and properly imported
        chdir_results = chdir(data_np, sampleclass, self.X.columns,gamma=0.001)


        # Extract the top N features based on the CHDIR results
        top_features = chdir_results[:num_features]

        # Prepare the feature_importances DataFrame
        self.feature_importances = pd.DataFrame(top_features, columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True).reset_index(drop=True)    
    
    
    def train_fisher_score(self, num_features=20):
        

        # Calculate Fisher Score for each feature. Assuming self.X and self.y_encoded are correctly prepared
        scores = fisher_score.fisher_score(self.X.to_numpy(), self.y_encoded)

        # Rank features based on score and select the top ones
        top_features_indices = np.argsort(scores)[::-1][:num_features]
        top_features_names = self.X.columns[top_features_indices].tolist()
        top_features_scores = scores[top_features_indices]

        # Update the feature_importances attribute
        self.feature_importances = pd.DataFrame({
            'Feature': top_features_names,
            'FisherScore': top_features_scores
        }).sort_values(by='FisherScore', ascending=False).reset_index(drop=True)
        
    def train_random_forest(self, n_estimators=100, **kwargs):
        self.model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
        self.model.fit(self.X_train, self.y_train)
        self.feature_importances = pd.DataFrame({'Feature': self.X.columns, 'Importance': self.model.feature_importances_}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    def train_ttest(self, k=10, **kwargs):
        selector = SelectKBest(score_func=f_classif, k=min(k, len(self.X.columns)) if k != -1 else 'all', **kwargs)
        selector.fit_transform(self.X_train, self.y_train)
        scores = pd.Series(selector.scores_, index=self.X.columns)
        self.feature_importances = pd.DataFrame({'Feature': scores.index, 'Importance': scores.values}).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        if k != -1:
            self.feature_importances = self.feature_importances.head(k)

    def train_mutual_info(self, k=10, **kwargs):
        mi_scores = mutual_info_classif(self.X_train, self.y_train, **kwargs)
        self.feature_importances = pd.DataFrame({'Feature': self.X.columns, 'Importance': mi_scores}).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        if k != -1:
            self.feature_importances = self.feature_importances.head(k)

    def logistic_regression_importance(self, n=10, C=1.0, max_iter=500, **kwargs):
        """
        Train a logistic regression model and compute feature importances.
        
        Parameters:
        - n (int): Number of top features to return. If -1, returns all features.
        - C (float): Inverse of regularization strength; must be a positive float.
        - max_iter (int): Maximum number of iterations taken for the solvers to converge.
        - **kwargs: Additional keyword arguments to pass to LogisticRegression.
        """
        # Initialize and fit the logistic regression model
        model = LogisticRegression(C=C, max_iter=max_iter, **kwargs)
        model.fit(self.X_train, self.y_train)
        
        # Compute the importance of features
        importance = pd.Series(model.coef_[0], index=self.X.columns).abs()
        
        # Create a DataFrame to hold feature names and their importance scores
        feature_importances = pd.DataFrame({'Feature': importance.index, 'Importance': importance.values})
        
        # Sort features by their importance
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        # If n is not -1, return the top n features; otherwise, return all features
        if n != -1:
            self.feature_importances = feature_importances.head(n)
        else:
            self.feature_importances = feature_importances

    def train_decision_tree(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)
        self.model.fit(self.X_train, self.y_train)
        self.feature_importances = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)        

    def train_relief(self, n_features_to_select=10):
        if n_features_to_select == -1:
            n_features_to_select = len(self.X.columns)  # Select all features if -1 is specified
        
        relief = ReliefF(n_features_to_select=n_features_to_select,verbose=True, n_jobs=-1)
        relief.fit(self.X_train.values, self.y_train)
        feature_importances = relief.feature_importances_
        top_features_indices = feature_importances.argsort()[::-1][:n_features_to_select]  # This line now redundant but kept for clarity
        self.feature_importances = pd.DataFrame({'Feature': self.X.columns[top_features_indices], 'Importance': feature_importances[top_features_indices]}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # def wx(self,n_features_to_select):
    #     hp = WxHyperParameter(epochs=30, learning_ratio=0.01, batch_size=8, verbose=False)
    #     sel_idx, sel_weight, val_acc = wx_slp(x_train, y_train, x_val, y_val, n_selection=n_features_to_select, hyper_param=hp, num_cls=2)

    
    def train_mann_whitney_u(self):
        # Assuming binary classification and y_train is encoded as 0 and 1
        group1_indices = self.y_train == 0
        group2_indices = self.y_train == 1

        mwu_scores = []
        for feature in self.X.columns:
            group1 = self.X_train.loc[group1_indices, feature]
            group2 = self.X_train.loc[group2_indices, feature]
            stat, p = mannwhitneyu(group1, group2)
            mwu_scores.append(p)  # Using p-value to rank features. Lower p-value => higher significance

        # Ranking features by p-value (lower p-value means more significant difference between groups)
        self.feature_importances = pd.DataFrame({'Feature': self.X.columns, 'MWU p-value': mwu_scores}).sort_values(by='MWU p-value').reset_index(drop=True)

    def train_mrmr(self, k=10):
        # X = self.X_train.values
        # y = self.y_train
        if k == -1:
            k = len(self.X.columns)  # Assuming mrmr_classif handles -1 to return all features
        
        selected_features, relevance_scores, redundancy_scores = mrmr_classif(
            X=self.X_train, y=self.y_train, K=k,
            relevance='f',
            redundancy='c',
            return_scores=True,
            n_jobs=-1
        )
        absolute_redundancy_scores = redundancy_scores.abs()
        median_redundancy_per_feature = absolute_redundancy_scores.median(axis=1)
        median_redundancy_df = median_redundancy_per_feature.reset_index()
        median_redundancy_df.columns = ['Feature', 'Median_Redundancy_Score']

        # Display the first few rows of the dataframe
        # print(median_redundancy_df.head())
        relevance_df = relevance_scores.reset_index()
        relevance_df.columns = ['Feature', 'Relevance_Score']

        merged_df = pd.merge(relevance_df, median_redundancy_df, on='Feature')
        sorted_merged_df = merged_df.sort_values(by=['Median_Redundancy_Score', 'Relevance_Score'], ascending=[True, False])

        if k == -1:
            self.feature_importances =  sorted_merged_df  # Return all rows if k is -1
        else:
            self.feature_importances =  sorted_merged_df.head(k)  # Return top k rows


    def randomize_columns_order(self):
        """
        Randomizes the order of DataFrame columns, keeping the label column as the last column.
        """
        # Ensure the label column exists in the DataFrame
        if self.label_column not in self.dataframe.columns:
            raise ValueError("The DataFrame must contain the label column specified.")
        
        # Get a list of all columns except the label column
        columns_except_label = [col for col in self.dataframe.columns if col != self.label_column]
        
        # Shuffle the list of columns randomly
        np.random.shuffle(columns_except_label)
        
        # Append the label column back to the end of the list
        columns_random_order = columns_except_label + [self.label_column]
        
        # Reindex the DataFrame with the new order of columns
        self.dataframe = self.dataframe[columns_random_order]
        # Update X and y to reflect the new column order
        self.update_X_y()
        self.encode_labels()
        self.split_data()


    def train_wx(self, num_features=None, runs=10):
        """
        Performs feature selection using the WX algorithm on the class's data.

        Parameters:
        - num_features: Number of top features to select. If None, a default value should be used or determined by the algorithm.
        - runs: Number of runs for the WX algorithm. This parameter might be used internally by the WX algorithm.
        
        This method updates an attribute to store information about the selected features and their importance.
        """
        if num_features is None:
            num_features = 10  # Default or calculated value

        # Preparing data and labels
        genes = list(self.X.columns)
        data_labels = self.y_encoded
        y_all = to_categorical(data_labels, num_classes=np.unique(data_labels).size)
        x_all = self.X.values

        # Splitting data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=1)

        # WX algorithm hyperparameters setup
        hp = WxHyperParameter(epochs=30, learning_ratio=0.01, batch_size=8, verbose=False)
        sel_idx, sel_weight, val_acc = wx_slp(x_train, y_train, x_val, y_val, n_selection=num_features, hyper_param=hp, num_cls=np.unique(data_labels).size)

        # Updating the class attribute to reflect the selected features and their importance
        self.feature_importances = pd.DataFrame({
            'Feature': np.array(genes)[sel_idx],
            'Weight': sel_weight
        }).sort_values(by='Weight', ascending=False).reset_index(drop=True)

        # Optionally, you might want to log or use the validation accuracy somewhere
        print('\nSingle Layer WX Evaluation')
        print('Selected feature names:', np.array(genes)[sel_idx])
        print('Selected feature index:', sel_idx)
        print('Selected feature weight:', sel_weight)
        print('Evaluation accuracy:', val_acc)
        
if __name__ == '__main__':
    filename = "final_data.csv"
    df = GEODataManager.load_csv(filename)  # Adjust for your data loading function
    df['Label'] = df['Label'].replace({
    'Normal lung': 'normal',
    'airway': 'normal',
    'NSCLC': 'cancer',
    'Lung carcinoid tumor': 'cancer'
})
    
    normalizer = DataFrameNormalizer(df)
    df = normalizer.min_max_normalize('Label')
    normalizer.verify_dataset()

    columns_to_keep = ['Label'] + list(np.random.choice(df.columns[:-1], size=200, replace=False))

    # Creating the subset DataFrame
    df = df[columns_to_keep]
 
    selector = FeatureSelector(dataframe=df, label_column='Label')


    top_features_df = pd.DataFrame()

    print("Mutual Information ")
    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Applying Mutual Information with custom settings and returning all features
        selector.train_mrmr(k=-1)  # -1 to return all features
        # print(selector.feature_importances)
        # print(selector.X_train.columns)
        top_20_features = selector.feature_importances.head(20)
        top_20 = selector.feature_importances.nlargest(20, 'Relevance_Score')
        top_features_df[f'Iteration {i} Features'] = top_20['Feature'].reset_index(drop=True)
        top_features_df[f'Iteration {i} Relevance_Score'] = top_20['Relevance_Score'].reset_index(drop=True)
        top_features_df[f'Iteration {i} Redundancy_Score'] = top_20['Median_Redundancy_Score'].reset_index(drop=True)

    stability_calculator = StabilityCalculator(top_features_df)
    kenchev_stability_index = stability_calculator.calculate_kenchev_stability_index()
    print(f"Kenchev Stability Index: {kenchev_stability_index}")

    print("Chdir Score:")
    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Fine-tuning XGBoost and showing all features
        selector.train_chdir()
        # print(selector.feature_importances)  
        # print(selector.X_train.columns)
        top_20 = selector.feature_importances.head(20)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Scores - Iter {i}'] = top_20['Score'].reset_index(drop=True)  # Assume 'Score' is the column name
    stability_calculator = StabilityCalculator(top_features_df)
    kenchev_stability_index = stability_calculator.calculate_kenchev_stability_index()
    print(f"Kenchev Stability Index: {kenchev_stability_index}")


    top_features_df = pd.DataFrame()
    print("Wx Score:")
    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Fine-tuning XGBoost and showing all features
        selector.train_wx()
        # print(selector.feature_importances)  
        # print(selector.X_train.columns)
        top_20 = selector.feature_importances.head(20)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Scores - Iter {i}'] = top_20['Score'].reset_index(drop=True)  # Assume 'Score' is the column name

    stability_calculator = StabilityCalculator(top_features_df)
    kenchev_stability_index = stability_calculator.calculate_kenchev_stability_index()
    print(f"Kenchev Stability Index: {kenchev_stability_index}")





    top_features_df = pd.DataFrame()
    print("Fisher Score:")
    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Fine-tuning XGBoost and showing all features
        selector.train_fisher_score()
        # print(selector.feature_importances)  
        # print(selector.X_train.columns)
        top_20 = selector.feature_importances.head(20)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Scores - Iter {i}'] = top_20['Score'].reset_index(drop=True)  # Assume 'Score' is the column name
    stability_calculator = StabilityCalculator(top_features_df)
    kenchev_stability_index = stability_calculator.calculate_kenchev_stability_index()
    print(f"Kenchev Stability Index: {kenchev_stability_index}")
    
    top_features_df = pd.DataFrame()
    print("XGBoost:")
    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Fine-tuning XGBoost and showing all features
        xgboost_params = {'max_depth': 3, 'n_estimators': 100}
        selector.train_xgboost(**xgboost_params)
        # print(selector.feature_importances)  
        # print(selector.X_train.columns)
        top_20 = selector.feature_importances.head(20)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Scores - Iter {i}'] = top_20['Score'].reset_index(drop=True)  # Assume 'Score' is the column name
    stability_calculator = StabilityCalculator(top_features_df)
    kenchev_stability_index = stability_calculator.calculate_kenchev_stability_index()
    print(f"Kenchev Stability Index: {kenchev_stability_index}")

    top_features_df = pd.DataFrame()
    print("Decision Tree ")
    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Fine-tuning XGBoost and showing all features
        dt_params = {'max_depth':5}
        selector.train_decision_tree(**dt_params)
        # print(selector.feature_importances)  
        # print(selector.X_train.columns)
        top_20 = selector.feature_importances.head(20)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Scores - Iter {i}'] = top_20['Score'].reset_index(drop=True)  # Assume 'Score' is the column name
    stability_calculator = StabilityCalculator(top_features_df)
    kenchev_stability_index = stability_calculator.calculate_kenchev_stability_index()
    print(f"Kenchev Stability Index: {kenchev_stability_index}")




    top_features_df = pd.DataFrame()
    print("ANOVA F-test")

    for i in range(1, 6):  

        selector.randomize_columns_order()
        selector.train_ttest(k=-1)
        # print(selector.feature_importances)
        # print(selector.X_train.columns)
        top_20 = selector.feature_importances.head(20)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Scores - Iter {i}'] = top_20['Score'].reset_index(drop=True)  # Assume 'Score' is the column name    
    stability_calculator = StabilityCalculator(top_features_df)
    kenchev_stability_index = stability_calculator.calculate_kenchev_stability_index()
    print(f"Kenchev Stability Index: {kenchev_stability_index}")

    top_features_df = pd.DataFrame()
    print("Mann-Whitney U Test")

    for i in range(1, 6):  

        selector.randomize_columns_order()
        # Train using Mann-Whitney U test
        selector.train_mann_whitney_u()
        # print(selector.feature_importances)
        # print(selector.X_train.columns)
        top_20 = selector.feature_importances.head(20)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Scores - Iter {i}'] = top_20['Score'].reset_index(drop=True)  # Assume 'Score' is the column name

    stability_calculator = StabilityCalculator(top_features_df)
    kenchev_stability_index = stability_calculator.calculate_kenchev_stability_index()
    print(f"Kenchev Stability Index: {kenchev_stability_index}")

    top_features_df = pd.DataFrame()
    print("Logistic Regression")

    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Logistic Regression Importance with custom settings and returning all features
        lr_params = {'C': 0.01, 'max_iter': 1000}
        selector.logistic_regression_importance(n=-1, **lr_params)  # -1 to return all features
        # print(selector.feature_importances)
        # print(selector.X_train.columns)
        top_20 = selector.feature_importances.head(20)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Scores - Iter {i}'] = top_20['Score'].reset_index(drop=True)  # Assume 'Score' is the column name
    stability_calculator = StabilityCalculator(top_features_df)
    kenchev_stability_index = stability_calculator.calculate_kenchev_stability_index()
    print(f"Kenchev Stability Index: {kenchev_stability_index}")

    top_features_df = pd.DataFrame()
    print("ReliefF")

    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Train and select features using ReliefF
        selector.train_relief(n_features_to_select=-1)
        # print(selector.feature_importances)
        # print(selector.X_train.columns)
        top_20 = selector.feature_importances.head(20)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Scores - Iter {i}'] = top_20['Score'].reset_index(drop=True)  # Assume 'Score' is the column name
    stability_calculator = StabilityCalculator(top_features_df)
    kenchev_stability_index = stability_calculator.calculate_kenchev_stability_index()
    print(f"Kenchev Stability Index: {kenchev_stability_index}")


    print("END")
