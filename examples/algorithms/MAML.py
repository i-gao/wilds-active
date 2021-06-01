import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
import learn2learn as l2l
import torch.autograd.profiler as profiler

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

        print('INIT') 
        profile_cuda_memory()
    
    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
    
    def adapt_task(self, adapt_batch, eval_loader):
        """Clones meta_model -> task_model and trains task_model on train_data"""
        self.optimizer.zero_grad()

        print('START OF ADAPT_TASK') 
        profile_cuda_memory()

        # move meta_model to cpu, move model to cuda
        self.model = self.meta_model.clone()
        # self.model.to(self.device)
        torch.cuda.empty_cache()
        print(f">>>>>>>> Task model is on CUDA: {check_model_device(self.model, desire_cuda=True)}")
        print(f">>>>>>>> Meta model is on CUDA: {check_model_device(self.meta_model, desire_cuda=True)}")
       
      
        self._train_task_model(adapt_batch)
 
        results = self._eval_task_model(eval_loader, call_backward=True)


        # move meta_model to cuda, move model to cpu
        self.model.to('cpu')
        # self.meta_model.to(self.device)
        print(f">>>>>>>> Task model is on CPU: {check_model_device(self.model, desire_cuda=False)}")
        print(f">>>>>>>> Meta model is on CUDA: {check_model_device(self.meta_model, desire_cuda=True)}")
        
        self.optimizer.step()
        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)

        del self.model
        torch.cuda.empty_cache()

        print('END OF ADAPT_TASK') 
        profile_cuda_memory()
        
    def evaluate(self, adapt_data, eval_loader):
        self.optimizer.zero_grad()
        self.model = self.meta_model.clone()
        self._train_task_model(adapt_data)
        results = self._eval_task_model(eval_loader)
        results['objective'] = self.objective(results).item()

        self.update_log(results)
        return self.sanitize_dict(results)

    def _train_task_model(self, task_adaptation_data):
        self.model.train()
        for step in range(self.adaptation_steps):
            results = self.process_batch(task_adaptation_data)
            train_error = self.objective(results)
            self.model.adapt(train_error)
            train_error.detach()
        
        print('AFTER TRAIN TASK') 
        profile_cuda_memory()      
                
    def _eval_task_model(self, task_eval_loader, call_backward=False):
        self.model.eval()
        # Evaluate a trained task model on data from some loader
        epoch_y_true = []
        epoch_g = [] 
        epoch_metadata = []
        epoch_y_pred = []
        epoch_objective = 0

        with torch.set_grad_enabled(call_backward):
            for batch in task_eval_loader:
                print('BATCH') 
                profile_cuda_memory()

                batch_results = self.process_batch(batch)
                batch_objective = self.objective(batch_results)

                if call_backward: batch_objective.backward()  # do I retain graph here or not?

                print(">>>>>>>>>>>>> Meta model gradient norms: \n")
                print_model(self.meta_model)

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


def profile_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    print(f'>>>> CUDA Memory Status: total: {t}, allocated/needed: {a}, cached: {r-a}')

def print_model(model):
    out = {}
    for name, param in model.named_parameters():
        if param.grad is not None: out[name] = param.grad.norm().data
        else: out[name] = None
    print(out)

def check_model_device(model, desire_cuda=False):
    """make sure all params on this model are the desired device"""
    for param in model.parameters():
        if param.is_cuda != desire_cuda: return False
    return True
