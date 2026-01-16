#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/maed02/proj_repte4
#SBATCH -t 12-00:00
#SBATCH -p tfg
#SBATCH --mem 60000
#SBATCH -o /fhome/maed02/proj_repte4/lstm_lopo_%u_%j.out
#SBATCH -e /fhome/maed02/proj_repte4/lstm_lopo_%u_%j.err
#SBATCH --gres gpu:1

echo "========================================"
echo "🧠 LSTM-LOPO TEMPORAL - INICIO"
echo "========================================"
echo "Fecha: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

echo "Activant entorn virtual..."
source /fhome/maed02/proj_repte4/venv/bin/activate

cd /fhome/maed02/proj_repte4

echo ""
echo "📋 Configuración:"
echo "  - Modelo: LSTM (2 capas, hidden_size=128)"
echo "  - Pacientes test: chb10, chb04"
echo "  - Estrategia: LOPO con temporalidad preservada"
echo "  - Balance: Temporal undersample (sin shuffle)"
echo ""

echo "🚀 Iniciant LSTM-LOPO..."
python main_lstm_lopo.py

echo ""
echo "========================================"
echo "✅ LSTM-LOPO TEMPORAL - FINALITZAT"
echo "========================================"
echo "Fecha: $(date)"