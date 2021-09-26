dataset=waterbirds
K=50
selectby=None
sf=random
epochs=30
wd=0
gpus=1
unlabeled_weight=1
tau=0.8
seed=0
savepred=1
stepevery=2
pretrained=True
labeled_transform=None

#######

if [ $dataset == "fmow" ]; then
    lr=0.0001
    labeled=16
    unlabeled=56
elif [ $dataset == "waterbirds" ]; then
    lr=1e-5
    wd=1
    labeled=16
    unlabeled=56
elif [ $dataset == "celebA" ]; then
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

    if [ $labeled_transform == "None" ]; then
        _labeled_transform=""
    else
        _labeled_transform="--additional_labeled_transform ${labeled_transform}"
    fi

    if [ $setup == "consistency" ]; then
        algorithm="FixMatch"
        _setup_flag="--soft_pseudolabels --additional_unlabeled_transform None randaugment"
    elif [ $setup == "pseudolabel" ]; then
        algorithm="PseudoLabel"
        _setup_flag="--additional_unlabeled_transform None"
    elif [ $setup == "consistency+pseudolabel" ]; then
        algorithm="FixMatch"
        _setup_flag="--additional_unlabeled_transform None randaugment"
    elif [ $setup == "finetune" ]; then 
        algorithm="ERM"
        _setup_flag=""        
    fi

    _model_flag=""
    if [ $algorithm == "NoisyStudent" ]; then
        _model_flag="--teacher_model_path ${erm_path}"
    fi
    if [ $pretrained == "True" ]; then
        _model_flag="${_model_flag} --pretrained_model_path ${erm_path} "
    fi
    
    if [ $labels == "tgt" ]; then
        _label_args=""
    elif [ $labels == "src+tgt" ]; then
        _label_args="--use_source_labeled"
    elif [ $labels == "src+tgt-uniform" ]; then
        _label_args="--use_source_labeled --upsample_target_labeled"
    fi

    runstr="${setup}_${labels}-${sf}-${K}-by-${selectby}_lr=${lr}_wd=${wd}_labeled=${labeled}_unlabeled=${unlabeled}_seed=${seed}"

    dir="/juice/scr/irena/active-self-training/${dataset}/${runstr}"
    mkdir $dir

    #######

    cmd="CUDA_VISIBLE_DEVICES=$(seq -s, $counter 0 $(($gpus - 1))) python /juice/scr/irena/wilds-active/examples/run_expt.py --root_dir /u/scr/nlp/dro --device $(seq 0 $(($gpus - 1))) --loader_kwargs num_workers=$((4 * $gpus)) --log_dir $dir --dataset $dataset --lr $lr --weight_decay $wd --batch_size $labeled --unlabeled_batch_size $unlabeled --algorithm $algorithm $_model_flag --n_epochs $epochs --active_learning --selection_function $sf --n_shots $K $_selectby $_labeled_transform --eval_splits unlabeled_test unlabeled_test_disjoint id_test --save_splits unlabeled_test unlabeled_test_disjoint --self_training_unlabeled_weight $unlabeled_weight --self_training_threshold $tau $_label_args --seed $seed $_savepred --gradient_accumulation_steps $stepevery $_setup_flag --use_wandb --wandb_api_key_path /sailhome/irena/.ssh/wandb-api-key --wandb_kwargs entity=i-gao project=ssda group=batch name=$setup-$labels"

    nlp_cmd="nlprun -o ${dir}/out -n active --anaconda-environment wilds-ig --queue jag --priority standard --gpu-type titanxp --gpu-count $gpus --memory 16g \"$cmd\""

    #######

    echo $nlp_cmd > "${dir}/cmd"
    echo $nlp_cmd
    eval $nlp_cmd
}

######

for setup in "consistency" "pseudolabel" "consistency+pseudolabel" "finetune"
do
    for labels in "tgt" "src+tgt-uniform"
    do
        run_expt
    done
done
