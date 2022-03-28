import os, argparse
from subprocess import call
from sklearn.model_selection import ParameterGrid
from datetime import date

# any special experiment name to name output folders
EXP_NAME = "finetune-individual-cameras"
LOG_DIR = "/u/scr/irena/independent-iwildcam/individual-cameras"

# key is the argument name expected by main.py
# value should be a list of options
GRID_SEARCH = {
    'filter': [0] # train cameras
}

# key is the argument name expected by main.py
# value should be a single value
OTHER_ARGS = {
    'algorithm': "ERM",
    'dataset': "iwildcam",
    'lr': 3e-05,
    'weight_decay': 0,
    'n_epochs': 10,
    'pretrain_model_path': "/u/scr/irena/wilds-unlabeled-model-selection/models/0xc006392d35404899bf248d8f3dc8a8f2",
    'filterby_fields': ["location"],
    'filter_splits': ["train", "id_test", "id_val"],
    'val_split': "id_val",
    'evaluate_all_splits': False,
    'eval_splits': ["test", "id_test"],
    'use_wandb': True,
    'wandb_api_key_path': "/sailhome/irena/.ssh/wandb-api-key",
    'root_dir': "/u/scr/nlp/dro/wilds-datasets",
    'save_pred': True,
}

SLURM_ARGS = {
    'anaconda-environment': "wilds",
    'queue': "jag",
    'priority': "standard",
    'gpu-type': "titanxp",
    'gpu-count': 1,
    'memory': "12g",
}

def main(args):
    grid = ParameterGrid(GRID_SEARCH)
    input(f"This will result in {len(list(grid))} exps. OK?")

    name = f"_{EXP_NAME}" if EXP_NAME else ""
    for grid_params in grid:
        dirname = f"{LOG_DIR}/{date.today()}{name}"
        cmd = "python /u/scr/irena/wilds-active/examples/run_expt.py"

        # add grid search params
        for key, val in grid_params.items():
            dirname += f"_{key}={val}"
            cmd += f" --{key} {val}"
        
        cmd += f" --wandb_kwargs entity=i-gao project={EXP_NAME} group={grid_params['filter']}"

        # add other params
        for key, val in OTHER_ARGS.items():
            if isinstance(val, list): val = ' '.join(val)
            cmd += f" --{key} {val}"
        cmd += f" --log_dir {dirname}/log"

        # make dir
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # nlprun command
        nlprun_cmd = f'nlprun -n {name} -o {dirname}/out'
        for key, val in SLURM_ARGS.items():
            nlprun_cmd += f" --{key} {val}"
        nlprun_cmd += f' "{cmd}"'

        if args.print_only:
            print(nlprun_cmd)
        else:
            print(f"Running command: {nlprun_cmd}")
            call(nlprun_cmd, shell=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_only', action='store_true', default=False)
    args = parser.parse_args()
    main(args)