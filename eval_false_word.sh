#!/bin/sh
# Grid Engine options (lines prefixed with #$)

#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
#$ -N eval_false_word
#$ -cwd
#$ -o log/eval_false_word_out.log
#$ -e log/eval_false_word_err.log
#$ -l h_rt=24:00:00 
#$ -pe gpu 1
#$ -l h_vmem=256G

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
  --datasets maptask \
  --chunk_size 512 \
  --batch_size 2 \
  --false_word_ig