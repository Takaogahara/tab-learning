import pandas as pd
from pathlib import Path

from utils import ParametersHandler, DataHandler, IO
from processor import Dataprocessor
from model import Model, TestModel, explain_tabnet


def train(parameters):
    # Load dataset file
    df_train = pd.read_csv(parameters["DATA_TRAIN"], sep=",")

    # Split train data into train/test
    train, test = DataHandler.split_data(df_train, parameters, 0.8)
    X_train, y_train = train
    X_test, y_test = test

    col_feat = list(X_train.columns)
    parameters["DATA_COL_FEAT"] = col_feat

    # Process data
    processor = Dataprocessor(parameters)
    train_transform, cat_params, export = processor.fit_transform(X_train,
                                                                  y_train)

    parameters["DATA_OUTPUT"] = str(export)

    test_transform = processor.transform(X_test)

    ##########
    model = Model(cat_params, parameters)
    loss_dict = model.train((train_transform, y_train),
                            (test_transform, y_test))

    IO.export_yaml(parameters)
    return loss_dict


def test(parameters):
    # Get parameters
    parameters = ParametersHandler.test_parms(parameters)
    model_root = Path(parameters["DATA_OUTPUT"])

    # Load dataset file
    df_val = pd.read_csv(parameters["DATA_TEST"], sep=",")
    test_idx = df_val[parameters["DATA_COL_ID"]]
    X_valid = df_val.drop(columns=[parameters["DATA_COL_ID"]])

    # Process data
    processor = Dataprocessor(parameters)
    valid_transform = processor.transform(X_valid, load=model_root)

    ##########
    loader = TestModel(parameters)
    model = loader.load_tabnet()
    preds = model.predict(valid_transform.values)
    if parameters["DATA_SAVE_TABNET"]:
        explain_tabnet(model, valid_transform, model_root)

    IO.export_preds(test_idx, preds, parameters)
    return True
