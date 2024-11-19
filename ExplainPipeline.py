import torch
from xai.ShapExplainer import explain_beeswarm
from LiefMLP import load_model

def execute_pipeline(conf):
    explain_beeswarm(model = load_model(conf["model_path"]),
                     train_dataset = torch.load(conf["train_tensor_path"]),
                     bk_size = conf.getint("background_knowledge_size"),
                     test_dataset = torch.load(conf["test_tensor_path"]),
                     class_idx = conf.getint("class_index"),
                     flist_filepath = conf["features_list"],
                     shap_out_path = conf["expl_save_path"]
    )
    return