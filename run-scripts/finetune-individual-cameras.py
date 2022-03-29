import os, argparse
from subprocess import call
from sklearn.model_selection import ParameterGrid
from datetime import date

# any special experiment name to name output folders
EXP_NAME = "finetune-individual-cameras"
LOG_DIR = "/u/scr/irena/independent-iwildcam/individual-cameras"

TRAIN_LOCS = ['0','1','2','4','5','6','8','9','10','11','12','13','14','15','16','17','18','19','20','23','25','26','28','30','31','32','33','34','35','36','37','38','39','41','42','43','44','45','46','47','48','50','51','52','53','54','55','60','61','63','64','65','67','68','70','71','72','74','75','77','80','82','83','84','85','87','88','89','90','91','92','93','94','96','97','98','99','100','102','103','105','106','107','109','110','111','112','113','116','117','118','119','121','122','123','124','126','128','129','130','131','132','133','135','137','138','139','140','141','142','144','145','147','148','149','150','151','152','155','157','158','159','160','161','162','164','165','166','167','168','170','172','173','174','177','178','179','180','181','182','185','186','189','190','192','194','195','196','197','198','199','200','201','202','203','204','205','206','208','209','210','212','213','214','215','216','218','219','220','221','222','223','225','226','227','228','229','230','231','232','233','234','235','236','238','242','243','244','246','247','248','249','250','251','252','253','254','255','256','257','258','259','262','264','265','266','271','272','274','276','277','278','281','283','284','285','286','290','291','292','293','294','295','296','297','298','299','300','303','304','305','307','308','310','312','313','314','316','317','318','319','321','322']
TRAIN_CLUSTERS = [
    '0 13 33 71 92 113 122 123 164 173 186 200 220 236 281 291 298 303 310 316',
    '1 5 10 12 14 15 23 30 35 41 47 50 52 53 60 61 65 67 68 70 72 74 77 85 90 91 96 97 105 110 121 124 128 131 133 135 137 138 140 141 142 152 161 166 167 178 180 198 201 205 206 208 209 213 215 219 225 231 232 233 234 235 247 252 256 271 290 293 300 304 312 322',
    '2 6 26 48 54 63 84 102 117 118 160 162 168 179 189 221 223 230 243 249 253 255 259 262 286 294 295 296 307',
    '4 25 82 126 145 148 210 227',
    '42 64 100 246 250 278',
    '55',
]

# key is the argument name expected by main.py
# value should be a list of options
GRID_SEARCH = {
    'filter': TRAIN_LOCS, # train cameras
}

# key is the argument name expected by main.py
# value should be a single value
OTHER_ARGS = {
    'algorithm': "ERM",
    'dataset': "iwildcam",
    'lr': 1e-05,
    'weight_decay': 0,
    'n_epochs': 4,
    'pretrained_model_path': "/u/scr/irena/wilds-unlabeled-model-selection/models/0xc006392d35404899bf248d8f3dc8a8f2/best_model.pth",
    'filterby_fields': ["location"],
    'filter_splits': ["train", "id_test", "id_val"],
    'val_split': "id_val",
    'evaluate_all_splits': False,
    'eval_splits': ["test", "id_test"],
    'save_step': 2,
    'save_last': False,
    'log_every': 1,
    'use_wandb': True,
    'wandb_api_key_path': "/sailhome/irena/.ssh/wandb-api-key",
    'root_dir': "/u/scr/nlp/dro/wilds-datasets",
    'save_pred': True,
}

SLURM_ARGS = {
    'anaconda-environment': "wilds-ig",
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
        
        cmd += f" --wandb_kwargs entity=i-gao project={EXP_NAME} group=camera-{str(grid_params['filter'])}"

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
