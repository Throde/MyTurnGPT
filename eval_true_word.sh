#!/bin/sh
# Grid Engine options (lines prefixed with #$)

#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
#$ -N eval_true_word
#$ -cwd
#$ -o log/eval_true_word_out.log
#$ -e log/eval_true_word_err.log
#$ -l h_rt=48:00:00 
#$ -pe gpu-titanx 2
#$ -l h_vmem=64G

# priority
#$ -P lel_hcrc_cstr_students

# Email information
#$ -M s2125085@ed.ac.uk
#$ -m beas

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
#module load python/3.8  # 3.4.3
module load anaconda
source activate slptorch

# Run the program
# available datasets: taskmaster metalwoz multiwoz coached persona dailydialog
# python hello.py
python ./turngpt/eval.py \
  --model pretrained \
  --checkpoint turngpt/runs/TurnGPTpretrained/version_3/checkpoints/epoch=4-val_loss=2.37236.ckpt \
  --tokenizer turngpt/runs/TurnGPTpretrained/version_3/tokenizer.pt \
  --datasets maptask switchboard \
  --chunk_size 128 \
  --batch_size 1 \
  --num_workers 2 \
  --perplexity \
  --classification \
  --true_word_ig