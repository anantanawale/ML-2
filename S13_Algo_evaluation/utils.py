from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import pandas as pd


def get_classification_models():
    return [
        LogisticRegression(random_state=42),
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42, max_depth=5),
        HistGradientBoostingClassifier(random_state=42, max_depth=5),
        XGBClassifier(random_state=42, max_depth=5, n_estimators=200),
    ]


def evaluate_single_model(model, xtrain, ytrain, xtest, ytest):
    # Get 5 fold cross validation score for the model
    cv_scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring="f1_macro")
    # Calulate mean result of the cv scores
    cv_mean = cv_scores.mean()
    # Fit the model on train data
    model.fit(xtrain, ytrain)
    # Predict the results for train and test
    ypred_train = model.predict(xtrain)
    ypred_test = model.predict(xtest)
    # Calcule f1 scores
    f1_train = f1_score(ytrain, ypred_train, average="macro")
    f1_test = f1_score(ytest, ypred_test, average="macro")
    # Create a dictionary to save above results
    res = {
        "name": type(model).__name__,
        "model": model,
        "f1_cv": cv_mean,
        "f1_train": f1_train,
        "f1_test": f1_test,
    }
    return res


def evaluate_muliple_models(models: list, xtrain, ytrain, xtest, ytest):
    # This will store all the model results
    res = []
    # Apply for loop on the model objects
    for model in models:
        r = evaluate_single_model(model, xtrain, ytrain, xtest, ytest)
        print(r)
        print("="*50)
        res.append(r)
    # Show results in dataframe
    res_df = pd.DataFrame(res)
    # Sort the dataframe by cv scores
    sort_df = res_df.sort_values(by="f1_cv", ascending=False).reset_index(drop=True)
    # Get best model from above
    best_model = sort_df.loc[0, "model"]
    return best_model, sort_df