#!/bin/bash
# Set paths to your scripts and files
TRAINING_SCRIPT="src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 140 160 \
--epochs 180 --eps_end 0.03137 --dataset cifar10 --end_epoch_eps 82 \
--cert_reg bound_reg --eps_test 0.03137 --data_augmentation fast --lambda_ratio 0.7 \
--eps_test_L2 0.5 --eps_end_L2 0.5 --joint --lambda_ratio_L2 0.00001 --lp --max --reverse --lbd 2.0 --gp --beta 0.8"

cd ~/CURE

source ~/anaconda3/bin/activate
conda activate CURE
export PYTHONPATH=$PWD:$PYTHONPATH

# Step 1: Run the training script and capture the output
TRAINING_OUTPUT=$(python $TRAINING_SCRIPT)
# Step 2: Extract the model path from the training script's output
MODEL_PATH=$(echo "$TRAINING_OUTPUT" | grep "Model saved as:" | awk '{print $4}')
# Check if MODEL_PATH was successfully extracted
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path could not be found in the training script output."
    exit 1
fi
echo "Model path extracted: $MODEL_PATH"


cd ~/alpha-beta-CROWN/complete_verifier/
# conda deactivate
conda activate alpha-beta-crown

# Step 4: Run the analysis script
python abcrown.py --config exp_configs/my_test_l2_cifar_large.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l2_cifar_ours_large_0.8.txt
python abcrown.py --config exp_configs/my_test_l1_cifar_large.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l1_cifar_ours_large_0.8.txt
python abcrown.py --config exp_configs/my_test_linf_cifar_large.yaml --complete_verifier skip --share_alphas --onnx_path $MODEL_PATH > out_linf_cifar_ours_large_0.8.txt

# Step 5: calculate union accuracy
python cal_union.py --dataset cifar --file_path ours_large_0.8

echo "completed."