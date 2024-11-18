import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd


from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

seed = 42
np.random.seed(seed)

def tensor_to_numpy(ds):
    data = ds[:][0].numpy()
    label = ds[:][1].numpy()
    return [data, label]



def dataframe_to_numpy(df):
    data = df.drop(["filename", "sha256", "entropy", "label"], axis=1).to_numpy()
    label = df["label"].to_numpy()
    return [data, label]



def numpy_to_tensor_dataset(data, label, scaler = None):
    if scaler is not None:
        data = scaler.transform(data)

    t_data = torch.tensor(data, dtype=torch.float)
    t_label = torch.tensor(label, dtype=torch.long)

    ds = TensorDataset(t_data, t_label) 
    return ds



def numpy_to_dataframe(data, label, csv_out_path = None):
    df = pd.DataFrame(data)
    df["label"] = label

    if csv_out_path is not None:
        header = ["f"+str(i) for i in range(data.shape[1])].append("label")
        df.to_csv(csv_out_path, header=header, index=False)
        print("Dataset saved as csv")

    return df



def generate_adversary_dataset(dataset, eps, model, loss, input_shape = [1, 2381], nb_classes = 2, craft_all_samples = True, merge = True, return_tensors = False):
    if type(dataset) is TensorDataset:
        dataset = tensor_to_numpy(dataset)
    else:
        dataset[0] = dataset[0].astype(np.float32)
    classifier = PyTorchClassifier(model = model, loss = loss, input_shape = input_shape, nb_classes = nb_classes, device_type="gpu")
    crafter = FastGradientMethod(classifier, eps=eps)

    if not craft_all_samples:
        X_adv = crafter.generate(dataset[0][dataset[1] == 1])
        Y_adv = dataset[1][dataset[1] == 1]
    else:
        X_adv = crafter.generate(dataset[0])
        Y_adv = dataset[1]
    
    torch.cuda.empty_cache()
    if merge:
        X_adv = np.concatenate((dataset[0], X_adv))
        Y_adv = np.concatenate((dataset[1], Y_adv))
    
    if return_tensors:
        return TensorDataset(
            torch.tensor(X_adv, dtype=torch.float),
            torch.tensor(Y_adv, dtype=torch.long)
        )
    else:
        return X_adv, Y_adv



