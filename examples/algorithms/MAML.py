import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
import learn2learn as l2l

class MAML(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out).to(config.device)
        # self.meta_model is the main model that lasts
        # self.model is the task model that is frequently overwritten
        meta_model = l2l.algorithms.MAML(
            model, 
            lr=config.maml_adapt_lr, 
            first_order=config.maml_first_order
        ) 
        self.adaptation_steps = config.maml_n_adapt_steps
        
        # initialize module
        super().__init__(
            config=config,
            model=meta_model, # so that optim has correct params
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.meta_model = meta_model
        del self.model
    
    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
    
    def adapt_task(self, adapt_batch, eval_loader):
        """Clones meta_model -> task_model and trains task_model on train_data"""
        self.optimizer.zero_grad()
        self.model = self.meta_model.clone()
        self._train_task_model(adapt_batch)
        task_eval_error = self._eval_task_model(eval_loader)['objective']
        task_eval_error.backward()
        return task_eval_error
        
    def meta_update(self, meta_batch_size):
        """steps optimizer after adapting to tasks"""
        # Average the accumulated gradients and optimize
        for p in self.meta_model.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        self.optimizer.step()
        del self.model

    def evaluate(self, adapt_data, eval_loader):
        self.optimizer.zero_grad()
        self.model = self.meta_model.clone()
        self._train_task_model(adapt_data)
        results = self._eval_task_model(eval_loader)

        self.update_log(results)
        return self.sanitize_dict(results)

    def _train_task_model(self, task_adaptation_data):
        for step in range(self.adaptation_steps):
            if type(task_adaptation_data) == tuple:
                results = self.process_batch(task_adaptation_data)
            else:
                results = self.__loss_loop(task_adaptation_data)
            train_error = self.objective(results)
            self.model.adapt(train_error)    
                
    def _eval_task_model(self, task_eval_data):
        if type(task_eval_data) == tuple:     
            results = self.process_batch(task_eval_data)
        else:
            results = self.__loss_loop(task_eval_data)
        results['objective'] = self.objective(results)
        return results
        
    def __loss_loop(self, loader):
        # Evaluate a trained task model on data from some loader
        epoch_y_true = []
        epoch_g = [] 
        epoch_metadata = []
        epoch_y_pred = []
        for batch in loader:
            batch_results = self.process_batch(batch)
            epoch_y_true.append(batch_results['y_true'])
            epoch_g.append(batch_results['g'])
            epoch_metadata.append(batch_results['metadata'])
            epoch_y_pred.append(batch_results['y_pred'])
        return {
            'y_true': torch.cat(epoch_y_true),
            'g': torch.cat(epoch_g),
            'metadata': torch.cat(epoch_metadata),
            'y_pred': torch.cat(epoch_y_pred)
        }
