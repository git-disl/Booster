#!/bin/bash
#SBATCH -J smooth                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=20G
#SBATCH -o smooth_vaccine_agnews-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --exclude=atl1-1-01-006-19-0,atl1-1-01-006-29-0   # Exclude specific node
#SBATCH --account=gts-ll72-paid              # Tracking account
#SBATCH -q embers  


module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts

poison_ratio=0.05
bad_sample_num=5000
sample_num=1000
lamb=0.5
alpha=2 
model_path=${3:-meta-llama/Llama-2-7b-hf}   
path_after_slash=$(basename "$model_path") 
echo "The value of poison ratio is: $poison_ratio"
echo "The value of lamb is: $lamb"
echo "The value of sample number is: $sample_num"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory


# CUDA_VISIBLE_DEVICES=0 python train.py \
# 	--model_name_or_path ${model_path}\
# 	--lora_folder ckpt/${path_after_slash}_smooth_${alpha}_${lamb}_$bad_sample_num \
# 	--data_path PKU-Alignment/BeaverTails_dangerous \
# 	--bf16 True \
# 	--output_dir ckpt/agnews/${path_after_slash}_smooth_f_${alpha}_${lamb}_${poison_ratio}_${sample_num}_${bad_sample_num} \
# 	--num_train_epochs 20 \
# 	--per_device_train_batch_size 5 \
# 	--per_device_eval_batch_size 5 \
# 	--gradient_accumulation_steps 1 \
# 	--save_strategy "steps" \
# 	--save_steps 100000 \
# 	--save_total_limit 0 \
# 	--learning_rate 1e-5 \
# 	--weight_decay 0.1 \
# 	--warmup_ratio 0.1 \
# 	--lr_scheduler_type "constant" \
# 	--logging_steps 10 \
# 	--tf32 True \
# 	--eval_steps 2000 \
# 	--cache_dir cache \
# 	--optimizer normal \
# 	--evaluation_strategy  "steps" \
# 	--sample_num $sample_num \
# 	--poison_ratio ${poison_ratio} \
# 	--label_smoothing_factor  0 \
# 	--benign_dataset data/agnews.json \
# 	--bad_sample_num $bad_sample_num \
# 	--lamb ${lamb} \




cd poison/evaluation  


# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/${path_after_slash}_smooth_${alpha}_${lamb}_$bad_sample_num \
# 	--lora_folder2 ../../ckpt/agnews/${path_after_slash}_smooth_f_${alpha}_${lamb}_${poison_ratio}_${sample_num}_${bad_sample_num}\
# 	--model_folder ${model_path} \
# 	--output_path ../../data/poison/agnews/${path_after_slash}_smooth_f_${alpha}_${lamb}_${poison_ratio}_${sample_num}_${bad_sample_num}


# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/poison/agnews/${path_after_slash}_smooth_f_${alpha}_${lamb}_${poison_ratio}_${sample_num}_${bad_sample_num}



cd ../../agnews

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/${path_after_slash}_smooth_${alpha}_${lamb}_$bad_sample_num \
	--lora_folder2 ../ckpt/agnews/${path_after_slash}_smooth_f_${alpha}_${lamb}_${poison_ratio}_${sample_num}_${bad_sample_num} \
	--model_folder ${model_path} \
	--output_path ../data/agnews/${path_after_slash}_smooth_f_${alpha}_${lamb}_${poison_ratio}_${sample_num}_${bad_sample_num}