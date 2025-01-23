#!/bin/bash
# Set paths to your scripts and files
TRAINING_SCRIPT="src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 120 140 \
    --epochs 160 --eps_end 0.5 --dataset cifar10 --l1 1e-6 --end_epoch_eps 82 \
    --cert_reg bound_reg --eps_test 0.5 --data_augmentation fast \
    --lambda_ratio 0 --L2_attack --box_attack pgd_concrete"
# ANALYSIS_SCRIPT="analyse_model.py"
# YAML_FILE="config.yaml"
cd ~/SABR
source ~/anaconda3/bin/activate
source activate SABR
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
# echo "Model path extracted: $MODEL_PATH"
# # Step 3: Update the YAML file with the new model path
# # if [ -f "$YAML_FILE" ]; then
# #     sed -i "s|path: .*|path: $MODEL_PATH|" $YAML_FILE
# #     echo "YAML file $YAML_FILE updated with new model path."
# # else
# #     echo "Error: YAML file $YAML_FILE not found."
# #     exit 1
# # fi
# MODEL_PATH=/home/enyij2/SABR/models/sabr_models/cifar/cifar10__CNN7__bn_1_1__eps_0.5__lambda_1e-05__L2/cifar10__CNN7__bn_1_1__eps_0.5__lambda_1e-05__2024_05_08-14_59_40.onnx

cd ~/alpha-beta-CROWN/complete_verifier/
source ~/anaconda3/bin/activate
conda activate alpha-beta-crown

# Step 4: Run the analysis script
# python $ANALYSIS_SCRIPT
python abcrown.py --config exp_configs/my_test_l2_cifar_large.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l2_cifar_l2_0.5_pgd.txt

# python abcrown.py --config exp_configs/my_test_linf_cifar_large.yaml --complete_verifier skip --share_alphas --onnx_path $MODEL_PATH > out_linf_cifar_l2_0.5_1k.txt

# Step 5: calculate union accuracy
# python cal_union.py --dataset cifar --file_path l2_0.5_1k

echo "completed."