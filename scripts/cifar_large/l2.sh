#!/bin/bash
# Set paths to your scripts and files
TRAINING_SCRIPT="src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 120 140 \
    --epochs 160 --eps_end 0.5 --dataset cifar10 --l1 1e-6 --end_epoch_eps 82 \
    --cert_reg bound_reg --eps_test 0.5 --data_augmentation fast \
    --lambda_ratio 0.00001 --L2_attack"

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
source ~/anaconda3/bin/activate
conda activate alpha-beta-crown

# Step 4: Run the analysis script
python abcrown.py --config exp_configs/my_test_l2_cifar_large.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l2_cifar_l2_0.5.txt
python abcrown.py --config exp_configs/my_test_l1_cifar_large.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l1_cifar_l2_0.5.txt
python abcrown.py --config exp_configs/my_test_linf_cifar_large.yaml --complete_verifier skip --share_alphas --onnx_path $MODEL_PATH > out_linf_cifar_l2_0.5.txt

# Step 5: calculate union accuracy
python cal_union.py --dataset cifar --file_path l2_0.5

echo "completed."