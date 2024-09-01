#!/bin/bash
#SBATCH -J lisa                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=20G
#SBATCH -o LDIFS_ep_trackdrift-%j.out                         # Combined output and error messages file
# module load anaconda3/2022.05.0.1
# module load cuda/11.7.0-7sdye3
module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts

guide_data_num=10000
RHO=0.0001 
# density=$2
lr=1e-5
ep=${1:-20}
poison_ratio=0.2
sample_num=5000 
align_step=100   
finetune_step=900 
model_path=meta-llama/Llama-2-7b-hf   
path_after_slash=$(basename "$model_path") 
echo "The value of lr is: $lr"
echo "The value of RHO is: $RHO"
# echo "The value of density is: $density"
echo "The value of poison_ratio is: $poison_ratio"
echo "The value of sample number is: $sample_num"
echo "The align step is: $align_step"
echo "The finetune step is: $finetune_step"
echo "The model is: $model_path"
echo "Guide data num is: $guide_data_num"
cd  ../../                            # Change to working directory

CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path}\
	--lora_folder ckpt/${path_after_slash}_sft  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/sst2/${path_after_slash}_LDIFS_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep} \
	--num_train_epochs ${ep} \
	--per_device_train_batch_size 5 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate ${lr} \
	--weight_decay 0 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "constant" \
	--logging_steps 10 \
	--tf32 True \
	--eval_steps 5000 \
	--cache_dir cache \
	--optimizer LDIFS \
	--evaluation_strategy  "steps" \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/sst2.json \
	--rho ${RHO} \
	--track_embedding_drift True \
	--track_embedding_before_train True


cd poison/evaluation  


# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/${path_after_slash}_lisa_${RHO} \
# 	--model_folder ${model_path} \
# 	--output_path ../../data/pred/lisa_${RHO}

# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/pred/lisa_${RHO}



CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_sft  \
	--lora_folder2 ../../ckpt/sst2/${path_after_slash}_LDIFS_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/sst2/${path_after_slash}_LDIFS_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep}


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/sst2/${path_after_slash}_LDIFS_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep}



cd ../../sst2

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/${path_after_slash}_sft  \
	--lora_folder2 ../ckpt/sst2/${path_after_slash}_LDIFS_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep} \
	--model_folder ${model_path} \
	--output_path ../data/sst2/${path_after_slash}_LDIFS_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep}