import os
import pandas as pd
from ember.features import PEFeatureExtractor
import numpy as np
from tqdm import tqdm


def extract_features(exes_path, class_idx, out_path = None, ret_features = False):
    
    feat_extr = PEFeatureExtractor(feature_version=2)
    f_header = ["f%d" % i for i in range(2381)]

    out_features = []
    for f in tqdm(os.listdir(exes_path)):
        with open(os.path.join(exes_path, f), "rb") as file:
            pe_features = np.array(feat_extr.feature_vector(file.read()), dtype = np.float32)
            out_features.append(pe_features)
    
    df = pd.DataFrame(out_features, columns = f_header)
    #df["filename"] = os.listdir(exes_path)
    df["label"] = class_idx
    
    #check if out_path is not None
    if out_path is not None:
        df.to_csv(out_path, index = False)

    if ret_features:
        return df


def execute_pipeline(conf):
    goodware_df = extract_features(exes_path = conf["goodware_folder"], class_idx = 0, ret_features = True)
    malware_df = extract_features(exes_path = conf["malware_folder"], class_idx = 1, ret_features = True)

    dataset_df = pd.concat([goodware_df, malware_df])
    dataset_df.to_csv(conf["out_csv_path"], index = False)

    print("Done.")
