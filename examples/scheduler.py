from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, StepLR, CosineAnnealingLR

def initialize_scheduler(config, optimizer, n_train_steps):
    # construct schedulers
    if config.scheduler is None:
        return None
    elif config.scheduler == 'linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=n_train_steps,
            **config.scheduler_kwargs)
        step_every_batch = True
        use_metric = False
    elif config.scheduler == 'ReduceLROnPlateau':
        assert config.scheduler_metric_name, f'scheduler metric must be specified for {config.scheduler}'
        scheduler = ReduceLROnPlateau(
            optimizer,
            **config.scheduler_kwargs)
        step_every_batch = False
        use_metric = True
    elif config.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, **config.scheduler_kwargs)
        step_every_batch = False
        use_metric = False
    elif config.scheduler == 'FixMatchLR':
        scheduler = LambdaLR(
            optimizer,
            lambda x: (1.0 + 10 * float(x) / n_train_steps) ** -0.75
        )
        step_every_batch = True
        use_metric = False
    elif config.scheduler == 'CosineLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            n_train_steps,
            config.scheduler_kwargs['min_lr'] * config.lr
        )
        step_every_batch = True
        use_metric = False
    else:
        raise ValueError(f'Scheduler: {config.scheduler} not supported.')

    # add an step_every_batch field
    scheduler.step_every_batch = step_every_batch
    scheduler.use_metric = use_metric
    return scheduler

def step_scheduler(scheduler, metric=None):
    if isinstance(scheduler, ReduceLROnPlateau):
        assert metric is not None
        scheduler.step(metric)
    else:
        scheduler.step()
