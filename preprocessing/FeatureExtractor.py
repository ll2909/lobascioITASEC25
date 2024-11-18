import os
import pandas as pd
from ember.features import PEFeatureExtractor
import numpy as np
from tqdm import tqdm


def extract_features(exes_path, out_path, class_idx, ret_features = False):
    
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
    
    df.to_csv(out_path, index = False)
    print("Done.")

    if ret_features:
        return df


