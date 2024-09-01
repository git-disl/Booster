#!/bin/bash
#SBATCH -J undercover                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=5G
#SBATCH -o lisa_eraser_poison_ratio-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts

guide_data_num=10000
RHO=1 
# density=$2
poison_ratio=${1:-0.1}
sample_num=5000 
align_step=100   
finetune_step=900 

dense_ratio=0.1
bad_sample_num=2000

model_path=${3:-meta-llama/Llama-2-7b-hf}   
path_after_slash=$(basename "$model_path") 
echo "The value of poison ratio is: $poison_ratio"
echo "The value of dense ratio is: $dense_ratio"
echo "The value of sample number is: $sample_num"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory


CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path}\
	--lora_folder ckpt/${path_after_slash}_sft  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/sst2/${path_after_slash}_lisa_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num} \
	--num_train_epochs 20 \
	--per_device_train_batch_size 5 \
	--per_device_eval_batch_size 5 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate 1e-4 \
	--weight_decay 0.1  \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "constant" \
	--logging_steps 10 \
	--tf32 True \
	--eval_steps 5000 \
	--cache_dir cache \
	--optimizer lisa \
	--evaluation_strategy  "steps" \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/sst2.json \
	--rho ${RHO} \
	--alignment_step ${align_step} \
	--finetune_step ${finetune_step} \
	--guide_data_num ${guide_data_num}


CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path}  \
	--lora_folder ckpt/${path_after_slash}_sft  \
	--lora_folder2 ckpt/sst2/${path_after_slash}_lisa_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num} \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/sst2/${path_after_slash}_lisa_eraser_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num} \
	--num_train_epochs 0 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  1e-4 \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "constant" \
	--logging_steps 10 \
	--tf32 True \
	--cache_dir cache \
	--optimizer prune_afterfinetune \
	--poison_ratio 1 \
	--sample_num $bad_sample_num \
	--dense_ratio $dense_ratio \
	--benign_dataset data/sst2.json \


cd poison/evaluation  


CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_sft  \
	--lora_folder2 ../../ckpt/sst2/${path_after_slash}_lisa_eraser_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/sst2/${path_after_slash}_lisa_eraser_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/sst2/${path_after_slash}_lisa_eraser_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}



cd ../../sst2

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/${path_after_slash}_sft  \
	--lora_folder2 ../ckpt/sst2/${path_after_slash}_lisa_eraser_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num} \
	--model_folder ${model_path} \
	--output_path ../data/sst2/${path_after_slash}_lisa_eraser_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}