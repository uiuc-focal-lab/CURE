#!/bin/bash
# Set paths to your scripts and files
TRAINING_SCRIPT="src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 140 160 \
--epochs 180 --eps_end 0.03137 --dataset cifar10 --end_epoch_eps 82 \
--cert_reg bound_reg --eps_test 0.03137 --data_augmentation fast --lambda_ratio 0.7 \
--eps_test_L2 0.5 --eps_end_L2 0.5 --joint --lambda_ratio_L2 0.00001 --max"
# ANALYSIS_SCRIPT="analyse_model.py"
# YAML_FILE="config.yaml"
cd ~/SABR
source ~/anaconda3/bin/activate
# source /opt/anaconda/etc/profile.d/conda.sh
# conda deactivate
conda activate SABR
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
# Step 3: Update the YAML file with the new model path
# if [ -f "$YAML_FILE" ]; then
#     sed -i "s|path: .*|path: $MODEL_PATH|" $YAML_FILE
#     echo "YAML file $YAML_FILE updated with new model path."
# else
#     echo "Error: YAML file $YAML_FILE not found."
#     exit 1
# fi

cd ~/alpha-beta-CROWN/complete_verifier/
conda activate alpha-beta-crown

# Step 4: Run the analysis script
# python $ANALYSIS_SCRIPT
python abcrown.py --config exp_configs/my_test_l2_cifar_large.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l2_cifar_max_large.txt
python abcrown.py --config exp_configs/my_test_l1_cifar_large.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l1_cifar_max_large.txt
python abcrown.py --config exp_configs/my_test_linf_cifar_large.yaml --complete_verifier skip --share_alphas --onnx_path $MODEL_PATH > out_linf_cifar_max_large.txt

# Step 5: calculate union accuracy
python cal_union.py --dataset cifar --file_path max_large

echo "completed."