import pickle
from sklearn.preprocessing import QuantileTransformer

seed = 42

def fit_qt(dataset, file_out):
    # Fit the QuantileTransformer to the dataset
    print("Fitting Quantile Transformer")
    qt = QuantileTransformer(output_distribution="normal", random_state=seed, copy=False)
    qt.fit(dataset)
    # Save the fitted transformer to a file
    with open(file_out, 'wb') as f:
        pickle.dump(qt, f)
    
    print("Done")
    return qt


def transform_data(dataset, qt):
    # Transform the dataset using the saved QuantileTransformer
    print("Transforming data")
    transformed_data = qt.transform(dataset)
    print("Done")
    return transformed_data

