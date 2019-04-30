from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tone import create_pipeline, stars_to_sentiment
import pandas as pd
import numpy as np


def main():
    # load data
    df_learn = pd.read_csv('rozetka_learn.csv')
    df_test = pd.read_csv('rozetka_test.csv')
    X_learn = df_learn['review'].values
    X_test = df_test['review'].values
    y_learn = stars_to_sentiment(df_learn['stars'].values)
    y_test = stars_to_sentiment(df_test['stars'].values)

    # learn
    model = create_pipeline()
    model.fit(X_learn, y_learn)

    # estimate
    y_pred = model.predict(X_test)
    score_recall = recall_score(y_test, y_pred, average=None)
    score_accuracy = accuracy_score(y_test, y_pred)
    score_precision = precision_score(y_test, y_pred, average=None)
    score_f1 = f1_score(y_test, y_pred, average=None)

    print(f"Precision score = {score_precision}, mean = {np.mean(score_precision)}")
    print(f" Accuracy score = {score_accuracy}")
    print(f"   Recall score = {score_recall}, mean = {np.mean(score_recall)}")
    print(f"       F1 score = {score_f1}, mean = {np.mean(score_f1)}")


if __name__ == '__main__':
    main()
