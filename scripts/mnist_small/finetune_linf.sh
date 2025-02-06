#!/bin/bash
# Set paths to your scripts and files
MODEL_PATH='your pretrained model path'
TRAINING_SCRIPT="src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 10  12 \
--epochs 14 --eps_end 0.1 --dataset mnist --l1 1e-5 --end_epoch_eps 22 \
--cert_reg bound_reg --bs 256 --lambda_ratio 0.4 --eps_test 0.1 \
--eps_test_L2 0.5 --eps_end_L2 0.5 --joint --lambda_ratio_L2 0.00001 --max --lp --lbd 0.5 --start_epoch 0 \
--saved_net $MODEL_PATH"

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


cd ~/alpha-beta-CROWN/complete_verifier/
conda activate alpha-beta-crown

# Step 4: Run the analysis script
python abcrown.py --config exp_configs/my_test_l2_small.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l2_mnist_linf_finetune.txt
python abcrown.py --config exp_configs/my_test_l1_small.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l1_mnist_linf_finetune.txt
python abcrown.py --config exp_configs/my_test_linf_small.yaml --complete_verifier skip --share_alphas --onnx_path $MODEL_PATH > out_linf_mnist_linf_finetune.txt

# Step 5: calculate union accuracy
python cal_union.py --dataset mnist --file_path linf_finetune

echo "completed."
