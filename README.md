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
