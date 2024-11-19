import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from preprocessing.FeaturesPreprocessor import transform_data
from LiefMLP import load_model, test

def execute_pipeline(conf):

    if conf.getboolean("use_tensors"):
        # Load preprocessed tensors files
        test_ds = torch.load(conf["test_tensor_path"])
        adv_ds = torch.load(conf["adversary_tensor_path"])
    else:
        
        # Load pre-extracted features from CSV file
        test_features_df = pd.read_csv(conf["test_path"])
        adv_features_df = pd.read_csv(conf["adversary_path"])
        
        y_test = test_features_df.pop("label").to_numpy()
        x_test = test_features_df.to_numpy()
        y_adv = adv_features_df.pop("label").to_numpy()
        x_adv = adv_features_df.to_numpy()

        if conf.getboolean("qt_preprocess"):
            # Apply quantile transformation to the data
            f = open(conf["qt_path"], "rb")
            qt = pickle.load(f)
            f.close()
            print("Loaded Fitted Quantile Transformer") 
            x_test = transform_data(x_test, qt)
            x_adv = transform_data(x_adv, qt)
        
        test_ds = TensorDataset(
            torch.from_numpy(x_test).float(),
            torch.from_numpy(y_test).long()
        )
        adv_ds = TensorDataset(
            torch.from_numpy(x_adv).float(),
            torch.from_numpy(y_adv).long()
        )

     
    model = load_model(conf["model_path"])
    _, clf_report_m = test(model = model,
                                   test_loader = DataLoader(test_ds, batch_size = 1024, shuffle = True),
                                   criterion = torch.nn.CrossEntropyLoss(),
                                   plots_path = None,
                                   roc_plot = False,
                                   cm_plot = False)
    _, clf_report_a = test(model = model,
                                   test_loader = DataLoader(adv_ds, batch_size = 1024, shuffle = True),
                                   criterion = torch.nn.CrossEntropyLoss(),
                                   plots_path = None,
                                   roc_plot = False,
                                   cm_plot = False)

    return