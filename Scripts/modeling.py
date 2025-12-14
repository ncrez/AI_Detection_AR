# modeling.py

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def split_data(df):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, shuffle=True
    )

    return train_df, val_df, test_df

def build_tfidf(train_df, val_df, test_df, max_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)
    )

    vectorizer.fit(train_df["clean_text"])

    X_train = vectorizer.transform(train_df["clean_text"])
    X_val   = vectorizer.transform(val_df["clean_text"])
    X_test  = vectorizer.transform(test_df["clean_text"])

    return X_train, X_val, X_test

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC(kernel="linear", random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
