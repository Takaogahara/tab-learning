import pandas as pd

from utils import Utils
from processor import Dataprocessor
from model import Model


def export_submission(idx, preds, parameters):
    # out_path = parameters["DATA_OUTPUT"]
    col_id = parameters["DATA_COL_ID"]
    col_target = parameters["DATA_COL_TARGET"]
    df_submission = pd.DataFrame()

    df_submission[col_id] = idx
    df_submission[col_target] = preds

    out_path = ""
    df_submission.to_csv(out_path, sep=",", index=False)


def train(parameters):
    # Load dataset file
    df_train = Utils.load_file(parameters["DATA_TRAIN"])

    # Split train data into train/test
    train, test = Utils.split_data(df_train, parameters, 0.8)
    X_train, y_train = train
    X_test, y_test = test

    col_feat = list(X_train.columns)
    parameters["DATA_COL_FEAT"] = col_feat

    # Process data
    processor = Dataprocessor(parameters)
    train_transform, categoricals = processor.fit_transform(X_train, col_feat)
    export_path = processor.export()

    parameters["DATA_OUTPUT"] = str(export_path)

    # test_transform = processor.transform(X_test)
    test_transform, _ = processor.fit_transform(X_test, col_feat)

    ##########
    constructor = Model(categoricals, parameters)
    loss = constructor.train((train_transform, y_train),
                             (test_transform, y_test))
    # constructor.explain_tabnet(X_transform)

    Utils.export_yaml(parameters)
    return loss


def test(parameters, model, export_path):
    # Load dataset file
    df_val = Utils.load_file(parameters["DATA_TEST"])
    test_idx = df_val[parameters["DATA_COL_ID"]]
    X_valid = df_val.drop(columns=[parameters["DATA_COL_ID"]])

    # Process data
    processor = Dataprocessor(parameters)
    # valid_transform = processor.transform(X_valid, load=export_path)
    col_feat = list(X_valid.columns)
    valid_transform, _ = processor.fit_transform(X_valid, col_feat)

    ##########
    valid_transform = valid_transform.values
    preds = model.predict(valid_transform)

    export_submission(test_idx, preds, parameters)
