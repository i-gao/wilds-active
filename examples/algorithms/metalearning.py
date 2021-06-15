import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
import learn2learn as l2l

class MetaLearning(SingleModelAlgorithm): 
    def __init__(self, meta_model, config, grouper, loss,
            metric, n_train_steps):
        self.adaptation_steps = config.metalearning_n_adapt_steps
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
    
    def adapt_task(self, adapt_batch, eval_loader):
        """Clones meta_model -> task_model and trains task_model on train_data"""
        self.optimizer.zero_grad()
        self.model = self.meta_model.clone()
        torch.cuda.empty_cache()

        self._train_task_model(adapt_batch) 
        results = self._eval_task_model(eval_loader, call_backward=True)
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
        self.optimizer.zero_grad()
        self.model = self.meta_model.clone()
        self._train_task_model(adapt_data)
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
            self.model.adapt(train_error)
            train_error.detach() 
            print(f">>> Adapt step {step}: train_loss is {train_error.item()} (This should be decreasing)") # TODO: remove
                
    def _eval_task_model(self, task_eval_loader, call_backward=False):
        self.model.eval()
        # Evaluate a trained task model on data from some loader
        epoch_y_true = []
        epoch_g = [] 
        epoch_metadata = []
        epoch_y_pred = []

        with torch.set_grad_enabled(call_backward):
            for batch in task_eval_loader:
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
        # turn off gradients for all modules except final module (assumed to be the classifier)
        for p in model.parameters():
            p.requires_grad = False
        *_, classifier = model.modules()
        for p in classifier.parameters():
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