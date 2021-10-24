dataset=TerraIncognita
command=$1
data_dir=$2
gpu_id=$3

CUDA_VISIBLE_DEVICES=${gpu_id} \
python3 -m domainbed.scripts.sweep ${command}\
       --datasets ${dataset}\
       --algorithms ERM IRM VREx MMD RSC ARM CORAL SagNet GroupDRO Mixup MLDG DANN MTL ANDMask IGA ERDG\
       --data_dir ${data_dir}\
       --command_launcher local\
       --single_test_envs\
       --steps 5001\
       --holdout_fraction 0.1\
       --n_hparams 20\
       --n_trials 3\
       --skip_confirmation\
       --hparams "$(<sweep/${dataset}/hparams.json)"\
       --output_dir "sweep/${dataset}/outputs"