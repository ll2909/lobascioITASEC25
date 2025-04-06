import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import pickle
import pandas as pd
import time
import os

from LiefMLP import LiefMLP, train, test, load_model
from utils import EarlyStopping

from sklearn.preprocessing import MinMaxScaler

from hyperopt import hp, fmin, Trials, atpe
from functools import partial
from cf.GradientCounterfactuals import CounterfactualGenerator, get_features_ranges
#import mlflow

#mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
torch.serialization.add_safe_globals([TensorDataset])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_csv = True
epochs = 10
pretrain_params = {
    "lr" : 0.001,
    "batch_size" : 512,
    "epochs" : 150,
    "es_patience" : 5,
    "model_path" : "./models/LiefMLP_pretrained.pth"
}


def set_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_dataset(df : pd.DataFrame, scaler_path : str = None, save_path : str = "./", ds_filename : str = None):
    
    label = df.pop("label").to_numpy()
    data = df.to_numpy()
    
    if scaler_path is None:
        scaler = MinMaxScaler(feature_range=(0, 1), copy=False, clip=True)
        scaler.fit(data)
        pickle.dump(scaler, open(os.path.join(save_path, "minmax_scaler.pkl"), "wb"))
    
    else:
        scaler = pickle.load(open(scaler_path, "rb"))

    data = scaler.transform(data)

    ds = TensorDataset(
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(label, dtype=torch.long)
    )

    if ds_filename:
        torch.save(ds, os.path.join(save_path, ds_filename))

    
    return ds


def cf_objective(space ,cf_params, dataset):
    cf_gen = CounterfactualGenerator(
        model=cf_params["model"],
        target_class=cf_params["target_class"],
        pred_loss_type=cf_params["pred_loss_type"],
        dist_loss_type=cf_params["dist_loss_type"],
        mutable_features=cf_params["mutable_features"],
        feature_ranges=cf_params["feature_ranges"],
        lambda_p=float(space["lambda_p"]),
        lambda_d=float(space["lambda_d"]),
        learning_rate=float(space["learning_rate"]),
        max_iterations=25,
        early_stopping_patience=10,
    )
    _, _, report = cf_gen.generate_batch_counterfactuals(dataset, batch_size=cf_params["batch_size"])

    obj_success = 1 - report["success"]
    obj_sparsity = 1.0 if np.isnan(report["sparsity"]) else report["sparsity"]

    return (obj_success + obj_sparsity) / 2
    # Alternative: use only the success rate
    # return obj_success




def execute_pipeline():
    
    # 0: Load data and preprocess it (in case of csv)
    # or load the preprocessed TensorDataset
    if load_csv:
        train_ds = preprocess_dataset(pd.read_csv())
        valid_ds = preprocess_dataset(pd.read_csv(), scaler_path = "./minmax_scaler.pkl")
        test_set = preprocess_dataset(pd.read_csv(), scaler_path = "./minmax_scaler.pkl")
        adv_ds = preprocess_dataset(pd.read_csv(), scaler_path = "./minmax_scaler.pkl")
    else:
        train_ds = torch.load()
        valid_ds = torch.load()
        test_set = torch.load()
        adv_ds = torch.load()

    # 1: Initialize the model, the loss and the optimizer and pretrain the model
    model = LiefMLP().to(device)
    loss = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_params["lr"])
    train(
        model = model,
        train_loader = DataLoader(train_ds, batch_size=pretrain_params["batch_size"], shuffle=True),
        valid_loader = DataLoader(valid_ds, batch_size=pretrain_params["batch_size"], shuffle=True),
        criterion = loss,
        optimizer = optimizer,
        scheduler = None,
        epochs = pretrain_params["epochs"],
        show_validation_metrics = True,
        early_stopping = EarlyStopping(patience = pretrain_params["es_patience"], min_delta = 0.0),
        model_path = pretrain_params["model_path"],
        plot = False
    )

    # 1.1 Load the best model and test it on test set and adversarial set
    best_model = load_model(pretrain_params["model_path"]).to(device)
    _, test_report = test(
        model = best_model,
        test_loader = DataLoader(test_set, batch_size=pretrain_params["batch_size"], shuffle=True),
        criterion = loss,
        plot = False
    )
    _, adv_report = test(
        model = best_model,
        test_loader = DataLoader(adv_ds, batch_size=pretrain_params["batch_size"], shuffle = True),
        criterion = loss,
        plot = False
    )

    # 2: Use the best model for the optimal hyperparameters search for cf generation with hyperopt, 
    #    using the validation(?) set (the best config will be fixed for the entire experiment)
    cf_params = {
        'model' : best_model,
        'target_class' : 'opposite',
        'pred_loss_type' : 'l2',
        'diss_loss_type' : 'l1',
        'feature_ranges' : get_features_ranges(train_ds[:][0]),
        #'learning_rate' : 0.1,
        'batch_size' : 1024,
        'max_iters' : 25,
        'mutable_features' : [i for i in range(2381)],
        #'lambda_p' : 1.0,
        #'lambda_d' : 1.0,
        'target_confidence' : 0.5
    }

    space = {
        'learning_rate' : hp.uniform('learning_rate', 0.05, 0.25),
        'lambda_p' : hp.uniform('lambda_p', 0.25, 1.25),
        'lambda_d' : hp.uniform('lambda_d', 0.25, 1.25),
    }

    trials = Trials()
    best = fmin(
        fn=partial(cf_objective, cf_params = cf_params, dataset = valid_ds),
        space=space,
        algo=atpe.suggest,
        max_evals = 20,
        trials=trials
    )
    print(best)
    

    # 3: Initialize the optimizer and the new loss for the new model and
    #    Start the new training loop with the generation of cfs, their verification
    #    and the training of the model with the generated cfs
    new_model = load_model(pretrain_params["model_path"]).to(device)
    # Alternative: training new model from scratch
    #new_model = LiefMLP().to(device)
    optimizer = torch.optim.Adam(new_model.parameters(), lr = 0.001)

    for ep in range(epochs):
        cf_model = LiefMLP()
        cf_model.parameters = best_model.parameters

    return



if __name__ == "__main__":
    
    set_reproducibility(42)
    execute_pipeline()