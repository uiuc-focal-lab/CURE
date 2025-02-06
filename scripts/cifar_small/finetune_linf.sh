#!/bin/bash
# Set paths to your scripts and files
MODEL_PATH='your pretrained model path'
TRAINING_SCRIPT="python src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 24 28 \
--epochs 32 --eps_end 0.00784313725 --dataset cifar10 --l1 1e-6 --end_epoch_eps 16 \
--cert_reg bound_reg --eps_test 0.00784313725 --data_augmentation fast \
--lambda_ratio 0.1 --shrinking_ratio 0.4 --use_shrinking_box --start_epoch 0 \
--shrinking_relu_state cross --shrinking_method to_zero_shrinking_box \
--eps_test_L2 0.25 --eps_end_L2 0.25 --joint --lambda_ratio_L2 0.00001 --lp --gp --reverse --lbd 0.5\
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
echo "Model path extracted: $MODEL_PATH"

cd ~/alpha-beta-CROWN/complete_verifier/

conda deactivate
conda activate alpha-beta-crown

# Step 4: Run the analysis script
python abcrown.py --config exp_configs/my_test_l2_cifar_small.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l2_cifar_ours_finetune_small.txt
python abcrown.py --config exp_configs/my_test_l1_cifar_small.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l1_cifar_ours_finetune_small.txt
python abcrown.py --config exp_configs/my_test_linf_cifar_small.yaml --complete_verifier skip --share_alphas --onnx_path $MODEL_PATH > out_linf_cifar_ours_finetune_small.txt

# Step 5: calculate union accuracy
python cal_union.py --dataset cifar --file_path ours_finetune_small

echo "completed."