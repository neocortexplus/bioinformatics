import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from data_handler import GEODataManager


class FeatureSelector:
    def __init__(self, dataframe, label_column):
        self.dataframe = dataframe
        self.label_column = label_column
        self.X = self.dataframe.drop(label_column, axis=1)
        self.y = self.dataframe[label_column]
        self.model = None
        self.feature_importances = None

    def encode_labels(self):
        le = LabelEncoder()
        self.y_encoded = le.fit_transform(self.y)

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=test_size, random_state=random_state)

    def train_xgboost(self, **kwargs):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **kwargs)
        self.model.fit(self.X_train, self.y_train)
        self.feature_importances = pd.DataFrame({'Feature': self.X.columns, 'Importance': self.model.feature_importances_})

    def train_random_forest(self, n_estimators=100, **kwargs):
        self.model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
        self.model.fit(self.X_train, self.y_train)
        self.feature_importances = pd.DataFrame({'Feature': self.X.columns, 'Importance': self.model.feature_importances_})

    def apply_ttest(self, k=10):
        # Apply t-test to select k best features based on the training data
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(self.X_train, self.y_train)
        # Get the mask of selected features
        mask = selector.get_support(indices=True)
        # Map back to feature names and return
        selected_features = self.X.columns[mask]
        return selected_features

    def get_top_n_features(self, n=10):
        sorted_features = self.feature_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return sorted_features.head(n)

# Example usage
if __name__ == '__main__':
    filename = "final_data.csv"
    df = GEODataManager.load_csv(filename)
    selector = FeatureSelector(dataframe=df, label_column='Label')
    selector.encode_labels()
    selector.split_data()
    
    # Train XGBoost and print top 10 features
    selector.train_xgboost()
    print("Top 10 Features from XGBoost:")
    print(selector.get_top_n_features(n=10))
    
    # Train Random Forest and print top 10 features
    selector.train_random_forest()
    print("\nTop 10 Features from Random Forest:")
    print(selector.get_top_n_features(n=10))
    
    # Apply t-test and print top 10 features
    print("\nTop 10 Features from t-test:")
    print(selector.apply_ttest(k=10))
