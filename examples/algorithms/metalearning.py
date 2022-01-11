import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
import learn2learn as l2l
import numpy as np

def sample_metalearning_task(K, M, grouper, support_set, labeled_set=None):
    """ 
    Samples metalearning (metatraining) task: K examples from the task for training, and M examples from the task for evaluation
    Args: 
        - K -- number of labeled shots for adaptation 
        - M -- number of examples for evaluation after adaptation
        - grouper -- WILDSGrouper; defined groups represent tasks
        - support_set -- the WILDSDataset to sample tasks (groups) from
        - labeled_set -- (optional) restrict labeled values to come from this WILDSDataset
    """
    if labeled_set is None: labeled_set = support_set
    if grouper is None:
        # Sample random points from whatever groups
        adaptation_idx = np.random.choice(
            np.arange(len(labeled_set)),
            K, 
            replace=(len(labeled_set) < K)
        )
        evaluation_idx = np.random.choice(
            np.arange(len(support_set)),
            M, 
            replace=(len(support_set) < M)
        )
        task=None      
    else:
        # Note that for all intents and purposes, labeled_set = support_set
        # This split just handles the transductive case.
        labeled_groups = grouper.metadata_to_group(labeled_set.metadata_array)
        support_groups = grouper.metadata_to_group(support_set.metadata_array)
        
        # Sample a specific group to be the task
        task = np.random.choice(
            support_groups.unique().numpy(),
            1
        )
        # Sample (x,y | task)
        adaptation_idx = np.random.choice(
            np.arange(len(labeled_set))[labeled_groups == task],
            K, 
            replace=(len(labeled_set) < K)
        )
        evaluation_idx = np.random.choice(
            np.arange(len(support_set))[support_groups == task],
            M, 
            replace=(len(support_set) < M)
        )
    # collect tensors
    x, y, m = zip(*[labeled_set[i] for i in adaptation_idx])
    adaptation_batch = (torch.stack(x), torch.stack(y), torch.stack(m))
    x, y, m = zip(*[labeled_set[i] for i in evaluation_idx])
    evaluation_batch = (torch.stack(x), torch.stack(y), torch.stack(m))
    return task, adaptation_batch, evaluation_batch

class MetaLearning(SingleModelAlgorithm): 
    def __init__(self, meta_model, config, grouper, loss,
            metric, n_train_steps):
        self.adaptation_steps = config.metalearning_kwargs.get('n_adapt_steps')
        super().__init__(
            config=config,
            model=meta_model, # so that optim has correct params
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # self.meta_model is the main model that lasts
        # self.model is the task model that is deleted and overwritten (per call to adapt_task)
        self.meta_model = meta_model
        del self.model # remove the self.model pointer (ref through self.meta_model instead)

    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
    
    def adapt_task(self, adapt_batch, eval_batch):
        """Clones meta_model -> task_model and trains task_model on train_data"""
        self.optimizer.zero_grad()
        self.model = self.meta_model.clone()
        torch.cuda.empty_cache()

        self._train_task_model(adapt_batch)
        results = self._eval_task_model(eval_batch, call_backward=True)
        print(f">>> Evaluation: test_loss for this task is {self.objective(results).item()}") # TODO: remove

        self.optimizer.step()
        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)
        self.model.to('cpu')
        del self.model
        torch.cuda.empty_cache()
        
    def evaluate(self, adapt_data, eval_loader):
        self.train() # for adaptation
        self.optimizer.zero_grad()
        self.model = self.meta_model.clone()

        self._train_task_model(adapt_data)
        self.eval() # for evaluation
        results = self._eval_task_model(eval_loader)
        results['objective'] = self.objective(results).item()

        self.model.to('cpu')
        del self.model
        torch.cuda.empty_cache()

        self.update_log(results)
        return self.sanitize_dict(results)

    def _train_task_model(self, task_adaptation_data):
        self.model.train()
        for step in range(self.adaptation_steps):
            results = self.process_batch(task_adaptation_data)
            train_error = self.objective(results)
            self.model.adapt(train_error, allow_nograd=True)
            train_error.detach() 
            print(f">>> Adapt step {step}: train_loss is {train_error.item()} (This should be decreasing)") # TODO: remove
                
    def _eval_task_model(self, task_eval_data, call_backward=False):
        self.model.eval()
        if type(task_eval_data) is tuple:
            # Used during adapt_task (meta_training)
            results = self.process_batch(task_eval_data)
            objective = self.objective(results)
            if call_backward: objective.backward()
            return self.sanitize_dict(results)
        else:
            # Used during evaluation (meta_evaluation)
            epoch_y_true = []
            epoch_g = [] 
            epoch_metadata = []
            epoch_y_pred = []

            with torch.set_grad_enabled(call_backward):
                for batch in task_eval_data:
                    batch_results = self.process_batch(batch)
                    batch_objective = self.objective(batch_results)
                    if call_backward: batch_objective.backward()

                    epoch_y_true.append(batch_results['y_true'].detach().clone())
                    epoch_g.append(batch_results['g'].detach().clone())
                    epoch_metadata.append(batch_results['metadata'].detach().clone())
                    epoch_y_pred.append(batch_results['y_pred'].detach().clone())
            
            return {
                'y_true': torch.cat(epoch_y_true),
                'g': torch.cat(epoch_g),
                'metadata': torch.cat(epoch_metadata),
                'y_pred': torch.cat(epoch_y_pred)
            }


class MAML(MetaLearning):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out).to(config.device)
        meta_model = l2l.algorithms.MAML(
            model, 
            lr=config.metalearning_adapt_lr, 
            first_order=config.maml_first_order
        ) 
        # initialize module
        super().__init__(
            meta_model=meta_model,
            config=config,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

class ANIL(MetaLearning):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out).to(config.device)
        for p in model.parameters():
            p.requires_grad = False
        *_, last = model.modules()
        for p in last.parameters():
            p.requires_grad = True
        meta_model = l2l.algorithms.MAML(
            model, 
            lr=config.metalearning_adapt_lr
        ) 
        # initialize module
        super().__init__(
            meta_model=meta_model,
            config=config,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
