#!/usr/bin/env python

from dataset import load_data, preprocess_data, create_dataset

def prepare_data(df):
    TIME_STEPS = 30
    train, test = preprocess_data(df)
    X_train, y_train = create_dataset(
    train[['close']],
    train.close,
    TIME_STEPS
    )

    X_test, y_test = create_dataset(
    test[['close']],
    test.close,
    TIME_STEPS
    )

    print(X_train.shape)

df = load_data()
prepare_data(df)
