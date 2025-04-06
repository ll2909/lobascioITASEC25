import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Optional, Tuple, Union

import numpy as np
import time

"""
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False"
"""

torch.serialization.add_safe_globals([TensorDataset])


def get_features_ranges(ds: torch.Tensor):
    feature_ranges = {}
    for i in range(ds.shape[1]):
        feature_ranges[i] = (ds[:, i].min(), ds[:, i].max())
    return feature_ranges


class CounterfactualGenerator:
    def __init__(
            self, 
            model: nn.Module, 
            target_class: Union[int, List[int], str] = "opposite",
            pred_loss_type: str = "ce",
            dist_loss_type: str = "l1",
            mutable_features = None,
            feature_ranges: Optional[dict] = None,
            lambda_p = 1.0,
            lambda_d = 1.0,
            lambda_s = 0.0,
            learning_rate: float = 0.01,
            max_iterations: int = 1000,
            early_stopping_patience: int = 50,
            early_stopping_threshold: float = 1e-5,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.target_class = torch.tensor(target_class, dtype=torch.long).to(self.device) if type(target_class) is not str else target_class
        self.pred_loss_type = pred_loss_type
        self.dist_loss_type = dist_loss_type
        self.mutable_features = mutable_features
        self.feature_ranges = feature_ranges
        self.lambda_p = lambda_p
        self.lambda_d = lambda_d
        self.lambda_s = lambda_s
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold

        # Median Absolute Deviation for normalizing distance loss, inducing sparsity as per Wachter et al.
        # in https://arxiv.org/abs/1711.00399

        self.mad = torch.tensor(1.0).to(self.device)
        self.median = None


    def compute_mad(self, train_ds):
      
        # Calculate median for each feature
        self.median = torch.median(train_ds, dim=0).values
        
        # Calculate absolute deviations from median
        abs_dev = torch.abs(train_ds - self.median.unsqueeze(0))
        
        # Calculate MAD for each feature (median of absolute deviations)
        self.mad = torch.median(abs_dev, dim=0).values
        
        # Option A: Add a small value only in the case of zero MAD to avoid division by zero, the addition must be added only to zero values
        #self.mad[self.mad == 0] = 1e-8
        #self.mad = self.mad.to(self.device)

        # Option B: Replace zeros with ones to avoid division by zero
        # This handles features with no variation
        self.mad = torch.where(self.mad > 0, self.mad, torch.ones_like(self.mad)).to(self.device)
    

    def prediction_loss(self, prediction, target_label):
        # NOTE : l1/l2 loss are generally used for regression, whereas ce and hinge for classification

        if self.pred_loss_type == "ce":
            loss = nn.CrossEntropyLoss()
            return loss(prediction, target_label)
        
        elif self.pred_loss_type ==  "l1": # MAE Loss
            loss = nn.L1Loss()
            return loss(prediction, target_label.float())
        
        elif self.pred_loss_type ==  "l2": # MSE Loss, used in Wachter et al. yloss term 
            loss = nn.MSELoss()
            return loss(prediction, target_label.float())
        
        elif self.pred_loss_type ==  "hinge":
            # Hinge loss assumes outputs are logits and targets are -1 or 1
            loss = nn.HingeEmbeddingLoss()
            return loss(prediction, target_label)
        
        else:
            raise NotImplementedError(f"{self.pred_loss_type} prediction loss not implemented, available losses are: ce, l1, l2, hinge.")
        

    def distance_loss(self, original, counterfactual):
        
        if self.dist_loss_type == "l1": # MAE proximity loss
            return torch.mean(torch.abs(counterfactual - original) / self.mad)
        
        elif self.dist_loss_type == "l2": # MSE proximity loss
            return torch.mean(torch.pow(counterfactual - original, 2) / self.mad)
        
        else:
            raise NotImplementedError(f"{self.distance_loss_type} distance loss not implemented, available losses are: l1, l2.")


    def sparsity_loss(self, original, counterfactual, sigma = 1e-6):
        # Differentiable L0-norm approximation proposed by Cinà et al. in Sigma-Zero: Gradient Based Optimization of l0-norm Adversarial Examples
        d = torch.abs(counterfactual - original)
        loss = torch.sum(d.pow(2) / (d.pow(2) + sigma))

        return loss

    """
    def sparsity_loss(self, original, counterfactual, eps = 1e-6):
        # NOTE : This is a differentiable approximation of L0 norm (number of non-zero elements),
        #        eps is a smoothing regularization constant
        #        tanh(d/eps) is near 1 when features are changed, 0 otherwise
        
        d = torch.abs(counterfactual - original)
        #loss = torch.mean(torch.tanh(100 * d).pow(2))
        #loss = torch.sum(d.pow(2) / d.pow(2) + 1e-5)
        loss = torch.mean(torch.tanh(d / eps))
        return loss
    """

    def get_loss(self, prediction, target_label, query_sample, cf_sample, target_confidence):

        pred_loss = self.prediction_loss(prediction, target_label)
        if prediction[target_label] >= target_confidence:
            pred_loss = torch.tensor(0.0, device = self.device)

        dist_loss = self.distance_loss(query_sample, cf_sample)
        sparsity_loss = self.sparsity_loss(query_sample, cf_sample)

        #print("Pred loss: ", pred_loss)
        #print("Dist loss: ", dist_loss)
        #print("Sparsity loss: ", sparsity_loss)

        return self.lambda_p * pred_loss + self.lambda_d * dist_loss + self.lambda_s * sparsity_loss


    def apply_feature_mask(self, original_input, counterfactuals, feature_mask):
        
        with torch.no_grad():
            # Only modify mutable features
            counterfactuals.data = (original_input * (1 - feature_mask) + counterfactuals.data * feature_mask)
        
        return counterfactuals


    def apply_feature_bounds(self, counterfactuals):
        # Apply to all samples
        for idx, (min_val, max_val) in self.feature_ranges.items():
            min_val = min_val.to(counterfactuals.device)
            max_val = max_val.to(counterfactuals.device)
            counterfactuals.data[:, idx].clamp_(min_val, max_val) 
        
        return counterfactuals
    

    def generate_counterfactual(
            self,
            original_input: torch.Tensor,
            original_label: torch.Tensor,
            target_confidence: float = 0.5
        ) -> Tuple[torch.Tensor, dict]:
        
        batch_size = original_input.shape[0]

        # Create copies of inputs that require gradients
        counterfactuals = torch.tensor(original_input, device = self.device)
        counterfactuals.requires_grad = True

        # Create mask for mutable features
        feature_mask = torch.ones_like(original_input)
        if self.mutable_features is not None:
            feature_mask = torch.zeros_like(original_input)
            feature_mask[:, self.mutable_features] = 1.0  # Apply mask to all samples in the batch

        # Move all the necessary data to device
        original_input = original_input.to(self.device)
        feature_mask = feature_mask.to(self.device)

        # Initialize optimizer
        optimizer = torch.optim.Adam([counterfactuals], lr=self.learning_rate)

        # Keep track of each cf losses
        best_losses = [float('inf')] * batch_size
        best_counterfactuals = [None] * batch_size
        patience_counters = [0] * batch_size
        histories = [{'losses': [], 'predictions': []} for _ in range(batch_size)]

        # Setup target labels
        if self.target_class == "opposite":
            desired_outcome = []
            for l in original_label:
                if l == 0:
                    if self.pred_loss_type == "hinge":
                        desired_outcome.append(1)
                    else:
                        desired_outcome.append(1)
                else:
                    if self.pred_loss_type == "hinge":
                        desired_outcome.append(-1)
                    else:
                        desired_outcome.append(0)

            desired_outcome = torch.tensor(desired_outcome, dtype = torch.long, device = self.device)

        else:
            desired_outcome = self.target_class.repeat(batch_size).to(self.device)


        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            # Get model predictions for the counterfactuals
            if self.pred_loss_type == "hinge":
                # get logits
                pred_probs = self.model.get_embeddings(counterfactuals, layer_idx = -1).to(self.device)
            else:
                pred_probs = self.model(counterfactuals).to(self.device)

            losses = []
            for i in range(batch_size):
                # Get the loss for the current counterfactual
                loss = self.get_loss(pred_probs[i], desired_outcome[i], original_input[i], counterfactuals[i], target_confidence)

                losses.append(loss)
                # Store history for each sample
                histories[i]['losses'].append(loss.item())
                histories[i]['predictions'].append(pred_probs[i][desired_outcome[i]].item())

                # Early stopping check for each sample
                if loss < best_losses[i] - self.early_stopping_threshold:
                    best_losses[i] = loss.item()
                    best_counterfactuals[i] = counterfactuals[i].clone().detach()
                    patience_counters[i] = 0
                else:
                    patience_counters[i] += 1

                # If all samples have reached early stopping, break the loop
                if patience_counters[i] >= self.early_stopping_patience:
                    if all(counter >= self.early_stopping_patience for counter in patience_counters):
                        break
            
            # Calculate the average loss for the batch
            batch_loss = torch.stack(losses).mean()
            # Backward pass and optimization step
            batch_loss.backward()
            optimizer.step()

            # Apply feature mask and feature range constraints
            counterfactuals = self.apply_feature_mask(original_input, counterfactuals, feature_mask)
            counterfactuals = self.apply_feature_bounds(counterfactuals)

        optimization_info = {
            'final_losses': best_losses,
            'iterations': iteration + 1,
            'loss_histories': histories,
        }

        # transform back to original labels
        if self.pred_loss_type == "hinge":
            desired_outcome = torch.tensor([0 if i == -1 else 1 for i in desired_outcome], dtype=torch.long)

        return torch.stack(best_counterfactuals), desired_outcome, optimization_info
    

    def verify_counterfactual(
            self,
            original_input: torch.Tensor,
            counterfactual: torch.Tensor,
            cf_labels,      # the target/desired outcome
            pred_threshold : float = 0.5,
            return_avg = True
        ) -> dict:
        
        # Get predictions
        with torch.no_grad():
            original_input = original_input.to(self.device)
            counterfactual = counterfactual.to(self.device)
            orig_probs = self.model(original_input).squeeze(0)
            cf_probs = self.model(counterfactual).squeeze(0)

            print("Generating report")

            # In case of a single sample
            cf_probs = cf_probs.unsqueeze(0) if len(cf_probs.shape) == 1 else cf_probs
            orig_probs = orig_probs.unsqueeze(0) if len(orig_probs.shape) == 1 else orig_probs

            # Calculate metrics for each sample
            success = np.array([])
            confidence = np.array([])

            for i in range(cf_probs.shape[0]):
                cf_conf = torch.max(cf_probs[i])
                cf_pred = torch.argmax(cf_probs[i])
                orig_pred = torch.argmax(orig_probs[i])
                if cf_pred != orig_pred and cf_conf >= pred_threshold:
                    success = np.append(success, True)
                else:
                    success = np.append(success, False)
            confidence = np.append(confidence, cf_conf.item())

            #for i in range(cf_probs.shape[0]):
            #  success.append(cf_probs[i][cf_labels[i]] >= pred_threshold)
            #  confidence.append(cf_probs[i][cf_labels[i]])

            proximity = np.array([torch.abs(c - o).mean().item() for c, o in zip(counterfactual, original_input)])
            sparsity = np.array([(c != o).float().mean().item() for c, o in zip(counterfactual, original_input)])
            l1 = np.array([torch.norm(c - o, p = 1).item() for c, o in zip(counterfactual, original_input)])
            l2 = np.array([torch.norm(c - o, p = 2).item() for c, o in zip(counterfactual, original_input)])
            #cossim = [(nn.functional.cosine_similarity(c, o, dim = 0).item() + 1)/2  for c, o in zip(counterfactual, original_input)]
            n_valid_cfs = success.sum()

        return {
            'success': n_valid_cfs/len(success) if return_avg else success,
            #'confidence': confidence,
            'proximity': sum(proximity * success)/n_valid_cfs if return_avg else proximity,
            'sparsity': sum(sparsity * success)/n_valid_cfs if return_avg else proximity,
            'manhattan': sum(l1 * success)/n_valid_cfs if return_avg else l1,
            'euclidean': sum(l2 * success)/n_valid_cfs if return_avg else l2,
            #'cossim' : sum(cossim)/n_valid_cfs if return_avg else cossim,
            #'original_class' : [o.argmax().item() for o in orig_probs],
            #'counterfactual_class' : [c.argmax().item() for c in cf_probs]
        }
    
    
    def generate_batch_counterfactuals(self, dataset : TensorDataset, batch_size : int):
    
        start = time.time()
        cfs = torch.tensor([])
        cfs_labels = torch.tensor([])
        
        print("Samples: ", dataset[:][0].shape)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in dl:
            cf_batch, cf_targets, _ = self.generate_counterfactual(
                original_input = batch[0].to(self.device),
                original_label = batch[1].to(self.device),
                target_confidence = 0.5
            )

            cf_batch = cf_batch.detach()
            cf_batch = cf_batch.cpu()
            cfs = torch.cat((cfs, cf_batch)).float()
            cfs_labels = torch.cat((cfs_labels, cf_targets.cpu())).long()


            print(cfs.shape)
            print(cfs_labels.shape)


        report = self.verify_counterfactual(
            original_input = dataset[:][0],
            counterfactual= cfs,
            cf_labels = cfs_labels,
        )
        
        orig_mean = dataset[:][0].mean().item()
        orig_std = dataset[:][0].std().item()
        cfs_mean = cfs.mean().item()
        cfs_std = cfs.std().item()
        exec_time = time.time() - start

        print("Counterfactuals generated in %i seconds" % exec_time)
        print("Orig. data Mean: %f\t Orig. data Std: %f" % (orig_mean, orig_std))
        print("CF Mean: %f\t CF Std: %f" % (cfs_mean, cfs_std))
        print("CF Generation success rate: \t%f" % report["success"])
        print("Avg Proximity: \t%f" % report["proximity"])
        print("Avg Sparsity: \t%f" % report["sparsity"])
        print("Avg Manhattan Distance: \t%f" % report["manhattan"])
        print("Avg Euclidean Distance: \t%f" % report["euclidean"])
        #print("Avg Cosine Similarity (Normalized): \t%f" % report["cossim"])
        print("\n")

        report["orig_mean"] = orig_mean
        report["orig_std"] = orig_std
        report["cf_mean"] = cfs_mean
        report["cf_std"] = cfs_std
        report["exec_time"] = exec_time

        return cfs, cfs_labels, report