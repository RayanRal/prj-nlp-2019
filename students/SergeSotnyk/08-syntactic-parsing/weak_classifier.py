from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile

from conllu_utils import load_trees, TRAIN_FILENAME, extract_features_and_y, TEST_FILENAME


def extract_x_y(name: str):
    trees = load_trees(name)
    x, y = [], []
    for tree in trees:
        xs, ys = extract_features_and_y(tree)
        x += xs
        y += ys
    return x, y


def store_weak_model(pipeline):
    joblib.dump(pipeline, "data/weak_classifier.sav", compress=True)


def load_weak_model():
    pipeline = joblib.load("data/weak_classifier.sav")
    return pipeline


def construct_pipeline():
    # How to predict probabilities in pipeline
    # https://stackoverflow.com/questions/42542975/computing-pipeline-logistic-regression-predict-proba-in-sklearn
    pipeline = Pipeline([
        ('dict_vectorizer', DictVectorizer()),
        ('selector', SelectPercentile(percentile=50)),
        ('classifier', LogisticRegression(random_state=1974, solver="newton-cg", max_iter=10000, verbose=1)),
    ])
    return pipeline


def main():
    x_train, y_train = extract_x_y(TRAIN_FILENAME)

    pipeline = construct_pipeline()
    pipeline.fit(x_train, y_train)
    vec = pipeline.named_steps['dict_vectorizer']
    print("\nTotal number of features: ", len(vec.get_feature_names()))

    x_test, y_test = extract_x_y(TEST_FILENAME)

    predicted = pipeline.predict(x_test)
    print(classification_report(y_test, predicted))

    store_weak_model(pipeline)


if __name__ == "__main__":
    main()
