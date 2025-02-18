import torch
import torch.nn.functional as F
from models.initializer import initialize_model
from algorithms.ERM import ERM
from algorithms.single_model_algorithm import SingleModelAlgorithm
from scheduler import LinearScheduleWithWarmupAndThreshold
from wilds.common.utils import split_into_groups
from configs.supported import process_outputs_functions
import copy
from utils import load

class PseudoLabel(SingleModelAlgorithm):
    """
    PseudoLabel.
    This is a vanilla pseudolabeling algorithm which updates the model per batch and incorporates a confidence threshold.

    Original paper:
        @inproceedings{lee2013pseudo,
            title={Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks},
            author={Lee, Dong-Hyun and others},
            booktitle={Workshop on challenges in representation learning, ICML},
            volume={3},
            number={2},
            pages={896},
            year={2013}
            }
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        model = initialize_model(config, d_out=d_out)
        model = model.to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # algorithm hyperparameters
        self.labeled_weight = config.self_training_labeled_weight
        self.unlabeled_weight_scheduler = LinearScheduleWithWarmupAndThreshold(
            max_value=config.self_training_unlabeled_weight,
            step_every_batch=True, # step per batch
            last_warmup_step=0,
            threshold_step=config.pseudolabel_T2*n_train_steps
        ) 
        self.schedulers.append(self.unlabeled_weight_scheduler)
        self.scheduler_metric_names.append(None)
        self.confidence_threshold = config.self_training_threshold
        if config.process_outputs_function is not None: 
            self.process_outputs_function = process_outputs_functions[config.process_outputs_function]
        # Additional logging
        self.logged_fields.append("pseudolabels_kept_frac")
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("consistency_loss")

    
    def process_batch(self, labeled_batch, unlabeled_batch=None):
        """
        Args:
            - labeled_batch: examples (x, y, m) 
            - unlabeled_batch: examples (x, m)
        Returns: results, a dict containing keys:
            - 'g': groups for the labeled batch
            - 'y_true': true labels for the labeled batch
            - 'y_pred': outputs (logits) for the labeled batch
            - 'metadata': metdata tensor for the labeled batch
            - 'unlabeled_g': groups for the unlabeled batch
            - 'unlabeled_y_pseudo': class pseudolabels of the unlabeled batch
            - 'unlabeled_mask': true if the unlabeled example had confidence above the threshold; we pass this around 
                to help compute the loss in self.objective()
            - 'unlabeled_y_pred': outputs (logits) on x of the unlabeled batch
            - 'unlabeled_metadata': metdata tensor for the unlabeled batch
        """
        assert labeled_batch is not None or unlabeled_batch is not None
        # Labeled examples
        x, y_true, metadata = labeled_batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'metadata': metadata
        }
        # Unlabeled examples
        if unlabeled_batch is not None:
            x_unlab, _, metadata = unlabeled_batch
            x_unlab = x_unlab.to(self.device)
            g = self.grouper.metadata_to_group(metadata).to(self.device)
            results['unlabeled_metadata'] = metadata
            results['unlabeled_g'] = g

        # Concat and call forward
        n_lab = x.shape[0]
        if unlabeled_batch is not None: x_concat = torch.cat((x, x_unlab), dim=0)
        else: x_concat = x

        outputs = self.model(x_concat)
        results['y_pred'] = outputs[:n_lab]

        if unlabeled_batch is not None:
            logits = outputs[n_lab:]
            results['unlabeled_y_pred'] = logits
            pseudo = logits.detach().clone()
            mask = torch.max(F.softmax(pseudo, -1), -1)[0] >= self.confidence_threshold
            pseudolabels = self.process_outputs_function(pseudo)
            results['unlabeled_y_pseudo'] = pseudolabels
            results['unlabeled_mask'] = mask

        return results
        
    def objective(self, results):
        # Labeled loss
        if 'y_pred' in results:
            classification_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        else:
            classification_loss = 0
        # Pseudolabeled loss
        if 'unlabeled_y_pred' in results:
            mask = results['unlabeled_mask']
            masked_loss_output = self.loss.compute_element_wise(
                results['unlabeled_y_pred'],
                results['unlabeled_y_pseudo'],
                return_dict=False,
            ) * mask
            consistency_loss = masked_loss_output.mean()
            pseudolabels_kept_frac = mask.count_nonzero().item() / mask.shape[0]
        else: 
            consistency_loss = 0
            pseudolabels_kept_frac = 0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", self.labeled_weight * classification_loss
        )
        self.save_metric_for_logging(
            results, "consistency_loss", self.unlabeled_weight_scheduler.value * consistency_loss
        )
        self.save_metric_for_logging(
            results, "pseudolabels_kept_frac", pseudolabels_kept_frac
        )

        return self.labeled_weight * classification_loss + self.unlabeled_weight_scheduler.value * consistency_loss 
