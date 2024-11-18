import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from preprocessing.FeatureExtractor import extract_features
from preprocessing.FeaturesPreprocessor import fit_qt, transform_data
from LIEF_FGSM import generate_adversary_dataset
from LiefMLP import load_model, train, LiefMLP
from utils.EarlyStopping import EarlyStopping


def execute_pipeline(conf):
    
    if conf.getboolean("use_tensors"):
        # Load preprocessed tensors files
        train_ds = torch.load(conf["train_tensor_path"])
        valid_ds = torch.load(conf["validation_tensor_path"])
    else:
        if conf.getboolean("extract_features"):
            # Extract features from the exes
            train_features_df = extract_features(conf["train_path"], conf["out_csv_path"]+"_train.csv", class_idx = 1, ret_features = True)
            valid_features_df = extract_features(conf["valid_path"], conf["out_csv_path"]+"_valid.csv", class_idx = 1, ret_features = True)
        else:
            # Load pre-extracted features from CSV file
            train_features_df = pd.read_csv(conf["train_path"])
            valid_features_df = pd.read_csv(conf["validation_path"])
        
        y_train = train_features_df.pop("label").to_numpy()
        x_train = train_features_df.to_numpy()
        y_valid = valid_features_df.pop("label").to_numpy()
        x_valid = valid_features_df.to_numpy()

        if conf.getboolean("qt_preprocess"):
            # Apply quantile transformation to the data
            if os.path.isfile(conf["qt_path"]):
                f = open(conf["qt_path"], "rb")
                qt = pickle.load(f)
                f.close()
                print("Loaded Fitted Quantile Transformer")
            else:
                qt = fit_qt(np.concatenate((x_train, x_valid)), conf["qt_path"])
            
            x_train = transform_data(x_train, qt)
            x_valid = transform_data(x_valid, qt)
        
        train_ds = TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).long()
        )
        valid_ds = TensorDataset(
            torch.from_numpy(x_valid).float(),
            torch.from_numpy(y_valid).long()
        )


    match conf.getint("at_fgsm"):
        case 1: # AdvFull mode
            print("FGSM Adversarial Training Mode: AdvFull")
            train_ds = generate_adversary_dataset(dataset = train_ds, 
                                                  eps = conf.getfloat("fgsm_epsilon"), 
                                                  model = load_model(conf["fgsm_base_model_path"]),
                                                  loss = torch.nn.CrossEntropyLoss(),
                                                  craft_all_samples = True,
                                                  merge = True,
                                                  return_tensors = True)
            valid_ds = generate_adversary_dataset(dataset = valid_ds, 
                                                  eps = conf.getfloat("fgsm_epsilon"), 
                                                  model = load_model(conf["fgsm_base_model_path"]),
                                                  loss = torch.nn.CrossEntropyLoss(),
                                                  craft_all_samples = True,
                                                  merge = True,
                                                  return_tensors = True)
        case 2: # AdvMal mode
            print("FGSM Adversarial Training Mode: AdvMal")
            train_ds = generate_adversary_dataset(dataset = train_ds, 
                                                  eps = conf.getfloat("fgsm_epsilon"), 
                                                  model = load_model(conf["fgsm_base_model_path"]),
                                                  loss = torch.nn.CrossEntropyLoss(),
                                                  craft_all_samples = False,
                                                  merge = True,
                                                  return_tensors = True)
            valid_ds = generate_adversary_dataset(dataset = valid_ds, 
                                                  eps = conf.getfloat("fgsm_epsilon"), 
                                                  model = load_model(conf["fgsm_base_model_path"]),
                                                  loss = torch.nn.CrossEntropyLoss(),
                                                  craft_all_samples = False,
                                                  merge = True,
                                                  return_tensors = True)
        case _:
            print("FGSM Adversarial Training not used.")

    model = LiefMLP()
    model = train(model = model,
                  train_loader = DataLoader(train_ds, batch_size = 1024, shuffle = True),
                  valid_loader = DataLoader(valid_ds, batch_size = 1024, shuffle = True),
                  criterion = torch.nn.CrossEntropyLoss(),
                  optimizer = torch.optim.Adam(model.parameters(), lr = 0.001),
                  scheduler = None,
                  n_epochs = 150,
                  show_validation_metrics = True,
                  early_stopping = EarlyStopping(patience = 5, min_delta = 0.00),
                  model_path = conf["model_path"],
                  plots_path = conf["plots_path"]
    )

    print("Training complete.")
            

    


