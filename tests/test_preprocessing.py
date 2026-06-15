import pandas as pd

from sklearn.model_selection import train_test_split
from src.features.preprocessing import build_preprocessor
from src.models.train_model import build_model, build_pipeline, train_pipeline
from src.config import RANDOM_STATE, TEST_SIZE


def test_build_preprocessor_splits_numeric_and_categorical_columns():
    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "score": [4.5, 7.0, 8.5],
            "gender": ["Male", "Female", "Male"],
            "country": ["US", "UK", "TR"],
        }
    )

    preprocessor = build_preprocessor(df)

    numeric_columns = preprocessor.transformers[0][2]
    categorical_columns = preprocessor.transformers[1][2]

    assert numeric_columns == ["age", "score"]
    assert categorical_columns == ["gender", "country"]


def test_pipeline_can_fit_and_predict_on_mixed_type_input():
    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "score": [4.5, 7.0, 8.5],
            "gender": ["Male", "Female", "Male"],
            "country": ["US", "UK", "TR"],
            "Target": [1, 0, 1],
        }
    )
    X = df.drop(columns="Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, _ = train_test_split(
        X, y, random_state=RANDOM_STATE, test_size=TEST_SIZE
    )
    preprocessor = build_preprocessor(X_train)
    model = build_model()
    pipeline = build_pipeline(preprocessor=preprocessor, model=model)
    train_pipeline(pipeline=pipeline, X_train=X_train, y_train=y_train)
    predictions = pipeline.predict(X_test)

    assert len(X_test) == len(predictions)
