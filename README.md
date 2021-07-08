Example Command: 

```
python /u/scr/irena/wilds-active/examples/run_expt.py --root_dir /u/scr/nlp/wilds/data --algorithm ERM --dataset camelyon17  --log_dir ./camelyon/random_$seed/ --load_dir ~/wilds-baselines/camelyon/official_erm/log --active_learning --selection_function random --n_shots 10 --n_rounds 3 --n_epochs 3 --evaluate_all_splits False --eval_splits unlabeled_test --few_shot_algorithm finetune --save_last --no_group_logging --seed $seed
```

Supported:
* Selection Functions: `random, uncertainty, uncertainty_fixed`
* FSL: `finetune, linear_probe`
