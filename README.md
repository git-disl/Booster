<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<h1 align="center">Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning</h1>



Lisa is a safety alignment method against thee threat of harmful fine-tuning. We consider a two-stage fine-tuning scheme: i) Alignment stage, in which we align the model with human-preference dataset (alignment dataset), and ii) finetuning stage, in which we finetune the model with a user finetuning dataset (which is mixed with harmful instance). Lisa is applied in the fine-tuning stage, in which a Bi-state optimization with proximal term is utilized to mitigate the risk of the mixed harmful data.     


## Main code logistic
We implement a cusomized trainer on top of the original HuggingFace Trainer. To achieve Bi-state optimization,  we append one line of code in function ` training_step()` of `trainer.py`. 

```
inputs = self.check_mode(inputs) //Appended code: switch dataset/model according to steps number
loss = step()  //Gradient backward with the data and model
```

To introduce a proximal term towards consensus, we need add the following regularization to the loss in function `step()`.

```
if self.status =="alignment":
  for name, param in model.named_parameters():
    if param.requires_grad and self.args.rho>0:
        loss +=  self.args.rho/2* torch.norm( param- self.alignment_weights[name])**2
else:
  for name, param in model.named_parameters():
    if param.requires_grad and self.args.rho>0:
         loss += self.args.rho/2* torch.norm( param- self.finetune_weights[name])**2
```
 



## Package requirement
The package requirement is listed in `lisa.yml` and `lisa_pip.txt`. Run the following code to install the packages with anaconda and pip.  
```
conda env create -f lisa.yml
pip install -r lisa_pip.txt
```

## Data  preparation
For finetuning task, we first need to run the following scripts to prepare the sueprvised finetuning data.
```
cd sst2
python build_dataset.py
cd ../gsm8k
python build_dataset.py
cd ../ag_news
python build_dataset.py
cd ..
```

## Huggingface Llama2 access
Llama2-7B is a gated repo, which need a formal request to get access to the model. Check out https://huggingface.co/meta-llama/Llama-2-7b-hf.
After applying permission from meta, you should be able to access the model, but you first need to enter your token in the file `huggingface_token.txt`.



## Example command to run

We prepare scripts for re-producing all the experiments in the paper. We recommend to use Slurm to reproduce the results as the logging file will be automatically organized into the script directory (if you don't use Slurm, just replace `sbatch` with `bash` in our example).

We first run SFT to produce the aligned model. 
```
cd script/alignment
sbatch  SFT.sh
```
Then we finetune the model using 10% of harmful data with a total number of 5000 samples from SST2 dataset. 
```
cd ../finetune
sbatch  lisa_poison_ratio.sh 0.1
```


For comparison, we finetune the model with SFT in the same data setting.

```
sbatch  sft_poison_ratio.sh 0.1
cd ../..
```





