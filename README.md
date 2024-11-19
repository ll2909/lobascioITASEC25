# lobascioITASEC25

The repository contains code refered to the work:

_Luca Lobascio, Giuseppina Andresini, Annalisa Appice, Donato Malerba_

[Adversarial Training to Improve Accuracy and Robustness of a Windows PE Malware Detection Model] (ITASEC2025)

Please cite our work if you find it useful for your research and work.
```
  @ARTICLE{, 
  author={L. {Lobascio} and G. {Andresini} and A. {Appice} and D. {Malerba}}, 
  journal={}, 
  title={Adversarial Training to Improve Accuracy and Robustness of a Windows PE Malware Detection Model}, 
  year={2025}, 
  volume={}, 
  number={}, 
  pages={},}
```

## Code Requirements

The code relies on the following **python3.6+** libs

Packages needed are:
* [PyTorch]
* [Numpy]
* [Pandas]
* [Scikit-learn]
* [LIEF]
* [ember]
* [SHAP]
* [Adversary-Robustness-Toolbox]

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
