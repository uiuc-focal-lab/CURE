#!/bin/bash
# Set paths to your scripts and files
TRAINING_SCRIPT="src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 50  60 \
--epochs 70 --eps_end 0.3 --dataset mnist --l1 1e-6 --end_epoch_eps 22 \
--cert_reg bound_reg --bs 256 --lambda_ratio 0.6 --eps_test 0.3 \
--eps_test_L2 1.5 --eps_end_L2 1.5 --joint --lambda_ratio_L2 0.00001"
# ANALYSIS_SCRIPT="analyse_model.py"
# YAML_FILE="config.yaml"
cd ~/SABR
# source /opt/anaconda/etc/profile.d/conda.sh
# conda deactivate
source ~/anaconda3/bin/activate
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
# MODEL_PATH=/home/enyij2/SABR/models/sabr_models/mnist/mnist__CNN7__bn_1_1__eps_0.3__lambda_0.6__lbd_1.5__avg/mnist__CNN7__bn_1_1__eps_0.3__lambda_0.6__lbd_1.5__2024_07_30-16_35_16.onnx
cd ~/alpha-beta-CROWN/complete_verifier/
source ~/anaconda3/bin/activate
conda activate alpha-beta-crown

# Step 4: Run the analysis script
# python $ANALYSIS_SCRIPT
python abcrown.py --config exp_configs/my_test_l2_large.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l2_mnist_joint_large.txt
python abcrown.py --config exp_configs/my_test_l1_large.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l1_mnist_joint_large.txt
python abcrown.py --config exp_configs/my_test_linf_large.yaml --complete_verifier skip --share_alphas --onnx_path $MODEL_PATH > out_linf_mnist_joint_large.txt

# Step 5: calculate union accuracy
python cal_union.py --dataset mnist --file_path joint_large

echo "completed."