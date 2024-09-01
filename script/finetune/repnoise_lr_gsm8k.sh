#!/bin/bash
#SBATCH -J sft                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=5G
#SBATCH -o repnoise_lr_gsm8k-%j.out                         # Combined output and error messages file

# module load anaconda3/2022.05.0.1
# module load cuda/11.7.0-7sdye3
module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts

# density=$2
lr=${1:-1e-4}
poison_ratio=0.2
alpha=0.1
beta=0.001
RHO=2
sample_num=5000 
model_path=meta-llama/Llama-2-7b-hf   
path_after_slash=$(basename "$model_path") 
# echo "The value of density is: $density"
echo "The value of poison ratio is: $poison_ratio"
echo "The value of dense ratio is: $dense_ratio"
echo "The value of sample number is: $sample_num"
echo "The learning rate is: $lr"
echo "alpha is: $alpha"
echo "beta is: $beta"
echo "The model is: $model_path"
cd  ../../                            # Change to working directory

CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path}\
	--lora_folder ckpt/${path_after_slash}_repnoise_${alpha}_${beta}  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_${lr} \
	--num_train_epochs 20 \
	--per_device_train_batch_size 5 \
	--per_device_eval_batch_size 5 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate ${lr} \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "constant" \
	--logging_steps 10 \
	--tf32 True \
	--eval_steps 2000 \
	--cache_dir cache \
	--optimizer normal \
	--evaluation_strategy  "steps" \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/gsm8k.json \


cd poison/evaluation  


# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/${path_after_slash}_sft_${RHO} \
# 	--model_folder ${model_path} \
# 	--output_path ../../data/pred/sft_${RHO}

# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/pred/sft_${RHO}



CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_repnoise_${alpha}_${beta} \
	--lora_folder2 ../../ckpt/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_${lr} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_${lr}


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_${lr}



cd ../../gsm8k

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/${path_after_slash}_repnoise_${alpha}_${beta} \
	--lora_folder2 ../ckpt/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_${lr} \
	--model_folder ${model_path} \
	--output_path ../data/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_${lr}