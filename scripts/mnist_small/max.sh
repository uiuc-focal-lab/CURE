#!/bin/bash
# Set paths to your scripts and files
TRAINING_SCRIPT="src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 50  60 \
--epochs 70 --eps_end 0.1 --dataset mnist --l1 1e-5 --end_epoch_eps 22 \
--cert_reg bound_reg --bs 256 --lambda_ratio 0.4 --eps_test 0.1 \
--eps_test_L2 0.75 --eps_end_L2 0.75 --joint --lambda_ratio_L2 0.00001 --max"
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

cd ~/alpha-beta-CROWN/complete_verifier/
conda deactivate
conda activate alpha-beta-crown

# Step 4: Run the analysis script
# python $ANALYSIS_SCRIPT
python abcrown.py --config exp_configs/my_test_l2_small.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l2_mnist_max_small.txt
python abcrown.py --config exp_configs/my_test_l1_small.yaml --complete_verifier skip --onnx_path $MODEL_PATH > out_l1_mnist_max_small.txt
python abcrown.py --config exp_configs/my_test_linf_small.yaml --complete_verifier skip --share_alphas --onnx_path $MODEL_PATH > out_linf_mnist_max_small.txt

# Step 5: calculate union accuracy
python cal_union.py --dataset mnist --file_path max_small

echo "completed."