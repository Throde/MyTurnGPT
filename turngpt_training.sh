#!/bin/sh
# Grid Engine options (lines prefixed with #$)
$ -N turngpt_traning              
$ -cwd /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2125085_Daohuan_Liu/MyTurnGPT
$ -o /home/s2125085/turngpt_job/output
$ -e /home/s2125085/turngpt_job/error
$ -l h_rt=00:25:00 
$ -l h_vmem=1G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
-P lel_hcrc_cstr_students
# Email information
-M s2125085@ed.ac.uk
-m beas

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load python/3.8  # 3.4.3

# Run the program
# available datasets: taskmaster metalwoz multiwoz coached persona dailydialog
# python hello.py
python ./turngpt/main.py \
  --model pretrained \
  --datasets maptask switchboard \
  --chunk_size 512 \
  --gpus 1 \
  --batch_size 2 \
