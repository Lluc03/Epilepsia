#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/maed02/proj_repte4
#SBATCH -t 12-00:00
#SBATCH -p tfg
#SBATCH --mem 60000
#SBATCH -o /fhome/maed02/proj_repte4/kfold_%u_%j.out
#SBATCH -e /fhome/maed02/proj_repte4/kfold_%u_%j.err
#SBATCH --gres gpu:1

echo "Activant entorn virtual..."
source /fhome/maed02/proj_repte4/venv/bin/activate

cd /fhome/maed02/proj_repte4
echo "Iniciant K-Fold LOPO..."
python main_baseline_kfolds.py

echo "K-Fold finalitzat."