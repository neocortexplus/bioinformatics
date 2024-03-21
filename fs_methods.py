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
            
    def train_relief(self, n_features_to_select=10):
        if n_features_to_select == -1:
            n_features_to_select = len(self.X.columns)  # Select all features if -1 is specified
        
        relief = ReliefF(n_features_to_select=n_features_to_select,verbose=True, n_jobs=-1)
        relief.fit(self.X_train.values, self.y_train)
        feature_importances = relief.feature_importances_
        top_features_indices = feature_importances.argsort()[::-1][:n_features_to_select]  # This line now redundant but kept for clarity
        self.feature_importances = pd.DataFrame({'Feature': self.X.columns[top_features_indices], 'Importance': feature_importances[top_features_indices]}).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    
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
        
        # If mrmr_classif doesn't return indices for all features when K=-1, adjust K accordingly before calling it
        self.feature_importances = pd.DataFrame({
            'Feature': self.X.columns[selected_features],
            'Relevance': relevance_scores,
            'Redundancy': redundancy_scores
        }).sort_values(by='Relevance', ascending=False).reset_index(drop=True)

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

def calculate_kenchev_stability_index(top_features_df):

    """

    Calculates the Kenchev Stability Index for feature selection results provided in a DataFrame.

    Each column of the DataFrame represents a set of top features selected in one iteration.

    

    Parameters:

    - top_features_df: DataFrame, where each column represents the top features selected in one iteration.

    

    Returns:

    - Kenchev Stability Index: A float value representing the stability of the feature selection process.

    """

    # Number of iterations (columns in the DataFrame)

    N = top_features_df.shape[1]

    

    # Number of top features selected in each iteration (rows in the DataFrame)

    k = top_features_df.shape[0]

    

    # Flatten the DataFrame to a list of lists (each sublist represents an iteration)

    features_all_iterations = [top_features_df[column].tolist() for column in top_features_df.columns]

    

    # Calculate pairwise overlaps and sum them up

    sum_overlaps = 0

    for i in range(N):

        for j in range(i + 1, N):

            sum_overlaps += len(set(features_all_iterations[i]).intersection(set(features_all_iterations[j])))



    # Compute Kenchev Stability Index

    K = sum_overlaps / (N * (N - 1) / 2 * k)

    

    return K
        
if __name__ == '__main__':
    filename = "../final_data.csv"
    df = GEODataManager.load_csv(filename)  # Adjust for your data loading function
    df['Label'] = df['Label'].replace({
    'Normal lung': 'normal',
    'airway': 'normal',
    'NSCLC': 'cancer',
    'Lung carcinoid tumor': 'cancer'
})
    # Keeping the 'Label' column (or the first column if it's not labeled) and randomly selecting 2000 features
    # Adjust the column selection if your label column has a different name or position
    columns_to_keep = ['Label'] + list(np.random.choice(df.columns[:-1], size=2000, replace=False))

    # Creating the subset DataFrame
    df = df[columns_to_keep]

    selector = FeatureSelector(dataframe=df, label_column='Label')

    top_features_df = pd.DataFrame()
    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Fine-tuning XGBoost and showing all features
        xgboost_params = {'max_depth': 3, 'n_estimators': 100}
        selector.train_xgboost(**xgboost_params)
        print("XGBoost Custom Parameters - All Features:")
        # print(selector.feature_importances)  
        # print(selector.X_train.columns)
        top_20_features = selector.feature_importances.head(20)['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20_features

    print(calculate_kenchev_stability_index(top_features_df))
    top_features_df = pd.DataFrame()
    for i in range(1, 6):  

        selector.randomize_columns_order()
        print("\nANOVA F-test Custom Parameters - All Features:")
        selector.train_ttest(k=-1)
        # print(selector.feature_importances)
        # print(selector.X_train.columns)
        top_20_features = selector.feature_importances.head(20)['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20_features
    
    print(calculate_kenchev_stability_index(top_features_df))
    top_features_df = pd.DataFrame()
    for i in range(1, 6):  

        selector.randomize_columns_order()
        # Train using Mann-Whitney U test
        selector.train_mann_whitney_u()
        print("\nMann-Whitney U Test Feature Ranking:")
        # print(selector.feature_importances)
        # print(selector.X_train.columns)
        top_20_features = selector.feature_importances.head(20)['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20_features


    print(calculate_kenchev_stability_index(top_features_df))
    top_features_df = pd.DataFrame()
    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Logistic Regression Importance with custom settings and returning all features
        lr_params = {'C': 0.01, 'max_iter': 1000}
        selector.logistic_regression_importance(n=-1, **lr_params)  # -1 to return all features
        print("\nLogistic Regression Importance Custom Parameters - All Features:")
        # print(selector.feature_importances)
        # print(selector.X_train.columns)
        top_20_features = selector.feature_importances.head(20)['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20_features

    print(calculate_kenchev_stability_index(top_features_df))
    top_features_df = pd.DataFrame()
    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Train and select features using ReliefF
        selector.train_relief(n_features_to_select=-1)
        print("\nReliefF Selected Features:")
        # print(selector.feature_importances)
        # print(selector.X_train.columns)
        top_20_features = selector.feature_importances.head(20)['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20_features

    print(calculate_kenchev_stability_index(top_features_df))
    top_features_df = pd.DataFrame()
    for i in range(1, 6):  
        selector.randomize_columns_order()
        # Applying Mutual Information with custom settings and returning all features
        selector.train_mrmr(k=-1)  # -1 to return all features
        print("\nMutual Information Custom Parameters - All Features:")
        # print(selector.feature_importances)
        # print(selector.X_train.columns)
        top_20_features = selector.feature_importances.head(20)['Feature'].reset_index(drop=True)
        top_features_df[f'Top 20 Features - Iter {i}'] = top_20_features

    print(calculate_kenchev_stability_index(top_features_df))
    print("END")
