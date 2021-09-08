dataset=fmow
selectby=region
algorithm=FixMatch
epochs=30
wd=0
gpus=1
K=50
unlabeled_weight=1
tau=0.8
seed=0
savepred=1
stepevery=2
pretrained=True

#######

if [ $dataset == "fmow" ]; then
    lr=0.0001
    labeled=16
    unlabeled=56
elif [ $dataset == "iwildcam" ]; then
    lr=3e-5
    labeled=16
    unlabeled=48
fi

#######

function run_expt() {
    if [ $selectby == "None" ]; then
        _selectby=""
    else
        _selectby="--selectby_fields ${selectby}"
    fi

    if [ $savepred == "None" ]; then
        _savepred=""
    else
        _savepred="--save_pred_step ${savepred} --save_pseudo_step ${savepred}"
    fi

    erm_path="/juice/scr/scr110/scr/nlp/wilds/baselines/${dataset}/official_erm/log/${dataset}_seed:0_epoch:best_model.pth"

    if [ $algorithm == "NoisyStudent" ]; then
        _model_flag="--teacher_model_path ${erm_path}"
    elif [ $pretrained == "True" ]; then
        _model_flag="--pretrained_model_path ${erm_path}"
    else
        _model_flag=""
    fi

    if [ $labels == "tgt" ]; then
        _label_args=""
        _label_runstr="${labels}-${sf}-${K}-by-${selectby}"
    elif [ $labels == "src+tgt" ]; then
        _label_args="--use_source_labeled"
        _label_runstr="${labels}-${sf}-${K}-by-${selectby}"
    elif [ $labels == "src+tgt-uniform" ]; then
        _label_args="--use_source_labeled --upsample_target_labeled"
        _label_runstr="${labels}-${sf}-${K}-by-${selectby}"
    elif [ $labels == "src" ]; then
        _label_args="--use_source_labeled --use_target_labeled False"
        _label_runstr="${labels}"
    elif [ $labels == "None" ]; then
        _label_args="--self_training_labeled_weight 0"
        _label_runstr="${labels}"
    fi

    runstr="${algorithm}_${_label_runstr}_lr=${lr}_wd=${wd}_labeled=${labeled}_unlabeled=${unlabeled}_seed=${seed}"

    dir="/juice/scr/irena/active-self-training/${dataset}/${runstr}"
    mkdir $dir

    #######

    cmd="CUDA_VISIBLE_DEVICES=$(seq -s, $counter 0 $(($gpus - 1))) python /juice/scr/irena/wilds-active/examples/run_expt.py --root_dir /u/scr/nlp/dro --device $(seq 0 $(($gpus - 1))) --loader_kwargs num_workers=$((4 * $gpus)) --log_dir $dir --dataset $dataset --lr $lr --weight_decay $wd --batch_size $labeled --unlabeled_batch_size $unlabeled --algorithm $algorithm $_model_flag --n_epochs $epochs --active_learning --selection_function $sf --n_shots $K $_selectby --eval_splits unlabeled_test unlabeled_test_disjoint id_test --save_splits unlabeled_test unlabeled_test_disjoint --self_training_unlabeled_weight $unlabeled_weight --self_training_threshold $tau $_label_args --seed $seed $_savepred --gradient_accumulation_steps $stepevery --use_wandb --wandb_api_key_path /sailhome/irena/.ssh/wandb-api-key --wandb_kwargs entity=i-gao project=ssda group=batch name=$labels"

    nlp_cmd="nlprun -o ${dir}/out -n active --anaconda-environment wilds-ig --queue jag --priority standard --gpu-type titanxp --gpu-count $gpus --memory 16g \"$cmd\""

    #######

    echo $nlp_cmd > "${dir}/cmd"
    echo $nlp_cmd
    eval $nlp_cmd
}

######

for labels in "None" "src" "tgt" "src+tgt-uniform"
do
    for sf in "random" 
    do
        run_expt
    done
done
