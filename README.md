# lobascioITASEC25

The repository contains code refered to the work:

_Luca Lobascio, Giuseppina Andresini, Annalisa Appice, Donato Malerba_

## Adversarial Training to Improve Accuracy and Robustness of a Windows PE Malware Detection Model
(Submitted to ITASEC2025)

Please cite our work if you find it useful for your research and work.
```
  @ARTICLE{, 
  author={L. {Lobascio} and G. {Andresini} and A. {Appice} and D. {Malerba}}, 
  journal={Submitted to ITASEC2025}, 
  title={Adversarial Training to Improve Accuracy and Robustness of a Windows PE Malware Detection Model}, 
  year={2025}, 
  volume={}, 
  number={}, 
  pages={},}
```

## Code Requirements

The code relies on the following **python3.6+** libs.

Packages needed are (exact version is advisable):
* PyTorch 2.0.1
* Numpy 1.25.2
* Pandas 2.0.3
* Scikit-learn 1.3.0
* LIEF 0.9.0
* ember 0.1.0
* SHAP 0.46.0
* Adversary-Robustness-Toolbox 1.17.1

## Data
The dataset used for experiments are accessible from the following links:
- [__BODMAS__](https://whyisyoung.github.io/BODMAS/)
- Dataset used in [__Windows-PE-Adversarial-Attacks__](https://github.com/MuhammdImran/Windows-PE-Adversarial-Attacks) by Muhammad Imran et al.

Pre-extracted features and preprocessed tensors are available at the following [__LINK__](https://unibari-my.sharepoint.com/:f:/g/personal/l_lobascio4_alumni_uniba_it/EllU1CnqXGZLqvPHxdhEHIIBbStAXmeSO7E_cNId4m8Meg?e=QeaCWz)

## How to use
Run the script main.py

The script can:
* Extract features from Windows PE executable files (.exe) and save to .csv file.
* Train the model using .csv files or preprocessed PyTorch TensorDataset files (.pt) as input, and preprocess the data using scikit-learn QuantileTransformer (only with .csv)
* Run inference tests using .csv files or preprocessed PyTorch TensorDataset files (.pt) as input
* Explain predictions using Shapley Values (SHAP) and visualize using the beeswarm plot. Note: works only with TensorDataset files (.pt)

To change settings, modify the file __config.conf__ according to the mode you're using.

## Config parameters

#### FEATURES_EXTRACTION
  - goodware_folder : path in which the goodware executables are sace.
  - malware_folder : path in which the malware executables are saved.
  - out_csv_path : where the .csv output file will be saved.

#### TRAIN
  - train_path : path to the .csv file for training data.
  - validation_path : path to the .csv file for validation data.
  - train_tensor_path : path to the .pt file for preprocessed tensors training data.
  - validation_tensor_path : path to the .pt file for preprocessed tensors validation data.
  - use_tensors : 1 if you want to used tensors files, 0 for using .csv files.
  - qt_preprocess : 1 if you want to preprocess data using scikit-learn QuantileTransformer, 0 otherwise (not advisable). This feature is available only if use_tensor is set to 0.
  - qt_path : where the QuantileTransformer file will be saved (or loaded, if it is already saved).
  - save_tensors : 1 if you want to save the preprocessed data as TensorDataset, 0 otherwise. This feature is available only if use_tensor_is set to 0. 
  - tensors_out_path : where the preprocessed .pt TensorDataset file will be saved, if save_tensors is set to 1.
  - at_fgsm : FGSM Adversarial Training Mode. 0 for regular training, 1 for generating adversarial samples for both classes (goodware and malware), 2 for generating adversarial samples only for malware class. The     aforementioned samples will be merged with the original training data.
  - fgsm_epsilon : the hyperparameter for adversarial samples generation.
  - fgsm_base_model_path : the path for the model FGSM will use to generate samples. You can first train a model without Adversarial Training (by setting at_fgsm to 0) and then use it for generating adverarial samples.
  - model_path : where the trained model file will be saved.
  - plots_path : where the plots images will be saved.

#### TEST
  - test_path : path to the .csv file for testing data.
  - adversary_path : path to the .csv file for adversary malwares testing data.
  - train_tensor_path : path to the .pt file for preprocessed tensors testing data.
  - validation_tensor_path : path to the .pt file for preprocessed tensors adversary malwares testing data.
  - use_tensors : 1 if you want to used tensors files, 0 for using .csv files.
  - qt_preprocess : 1 if you want to preprocess data using scikit-learn QuantileTransformer, 0 otherwise (not advisable). This feature is available only if use_tensor is set to 0.
  - qt_path : the path for loading the QuantileTransformer (you can fit the QT only in TRAIN).
  - save_tensors : 1 if you want to save the preprocessed data as TensorDataset, 0 otherwise. This feature is available only if use_tensor_is set to 0. 
  - tensors_out_path : where the preprocessed .pt TensorDataset file will be saved, if save_tensors is set to 1.
  - model_path : the path for loading the model file.
  - plots_path : where the plots images will be saved.

#### EXPLAIN
  - train_tensor_path : path to the .pt TensorDatset file for training data used as background knowledge.
  - test_tensor_path : path to the .pt TensorDataset file for testing data to explain.
  - model_path : the path for loading the model file to explain.
  - background_knowledge_size: number of samples used as background knowledge to generate explanations. A value higher than 1000 is computationally more expensive. It uses a stratified samples selection.
  - class_index : the class of the examples to explain. 0 for goodware, 1 for malware.
  - features_list : the features names list, used for the plot.
  - expl_save_path : where the the explanations files will be saved.
  - plots_path : where the plots images will be saved.
