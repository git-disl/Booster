#!/bin/bash
#SBATCH -J smooth                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=20G
#SBATCH -o smooth_statistic_evaluate_harmful-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts

poison_ratio=${1:-1}
bad_sample_num=5000
sample_num=1000  
alignment_sample_number=5000
lamb=5
alpha=0.1   
model_path=${3:-meta-llama/Llama-2-7b-hf}   
path_after_slash=$(basename "$model_path") 
echo "The value of poison ratio is: $poison_ratio"
echo "The value of lamb is: $lamb"
echo "The value of sample number is: $sample_num"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory




cd poison/evaluation  



steps=(200 400 600 800 1000 1200 1400 1600 1800 2000)

for step in "${steps[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py --input_path ../../harmful_score_steps_1.0_${step}
done
