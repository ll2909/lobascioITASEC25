import torch
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def explain_beeswarm(model, train_dataset, bk_size, test_dataset, class_idx, flist_filepath, shap_out_path = None, plot_out_path = None):
    f = open(flist_filepath, "r")
    flist = f.read().split(",")
    f.close()

    # Split the TensorDataset in data and label
    x_train = train_dataset[:][0].numpy()
    y_train = train_dataset[:][1].numpy()
    # Using a stratified sampling to obtain the background knowledge indexes
    _,bk_idx,_,_ = train_test_split(np.arange(len(x_train)), y_train, stratify=y_train,test_size=bk_size)
    sampled_train_ds = train_dataset[bk_idx][0]
    #print(torch.bincount(train_dataset[bk_idx][1]))
    
    # select from test datasets the tensors with label = class_idx
    samples_to_explain = test_dataset[test_dataset[:][1] == class_idx][0]

    explainer = shap.DeepExplainer(model.to(device), sampled_train_ds.to(device))

    start = time.time()
    print("Computing Shapley values")
    shap_values = explainer(samples_to_explain)
    print("Done in %s seconds" % (time.time() - start))
    shap_values.feature_names = flist

    if shap_out_path is not None:
        print("Saving Explanations.")
        f = open(shap_out_path, "wb")
        pickle.dump(shap_values, f)
        f.close()

    shap.plots.beeswarm(shap_values[:,:,class_idx], max_display = 21, show = False)
    if plot_out_path is not None:
        plt.savefig(plot_out_path, format = "png")
    plt.show()
    plt.clf()
    print("Done.")


