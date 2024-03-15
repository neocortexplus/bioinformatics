import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from data_handler import GEODataManager  # Make sure this import matches your setup

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

    def train_logistic_regression(self, n=10, C=1.0, max_iter=500, **kwargs):
        model = LogisticRegression(C=C, max_iter=max_iter, **kwargs)
        model.fit(self.X_train, self.y_train)
        importance = pd.Series(model.coef_[0], index=self.X.columns).abs()
        self.feature_importances = pd.DataFrame({'Feature': importance.index, 'Importance': importance.values}).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        if n != -1:
            self.feature_importances = self.feature_importances.head(n)


if __name__ == '__main__':
    filename = "final_data.csv"
    df = GEODataManager.load_csv(filename)  # Adjust for your data loading function
    selector = FeatureSelector(dataframe=df, label_column='Label')
    selector.encode_labels()
    selector.split_data()
    
    # Fine-tuning XGBoost and showing all features
    xgboost_params = {'max_depth': 3, 'n_estimators': 100}
    selector.train_xgboost(**xgboost_params)
    print("XGBoost Custom Parameters - All Features:")
    print(selector.feature_importances)  # Returns all features
    
    # Applying ANOVA F-test with custom settings and returning all features
    selector.apply_ttest(k=-1)  # -1 to return all features
    print("\nANOVA F-test Custom Parameters - All Features:")
    print(selector.feature_importances)
    
    # Applying Mutual Information with custom settings and returning all features
    mi_params = {'n_neighbors': 3}
    selector.apply_mutual_info(k=-1, **mi_params)  # -1 to return all features
    print("\nMutual Information Custom Parameters - All Features:")
    print(selector.feature_importances)
    
    # Logistic Regression Importance with custom settings and returning all features
    lr_params = {'C': 0.01, 'max_iter': 1000}
    selector.logistic_regression_importance(n=-1, **lr_params)  # -1 to return all features
    print("\nLogistic Regression Importance Custom Parameters - All Features:")
    print(selector.feature_importances)
    print("END")