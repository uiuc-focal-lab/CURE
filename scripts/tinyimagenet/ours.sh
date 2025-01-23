#!/bin/bash
# Set paths to your scripts and files
TRAINING_SCRIPT="src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 140 160 \
--epochs 180 --eps_end 0.00392157 --eps_test 0.00392157 --dataset tinyimagenet \
--end_epoch_eps 82 --cert_reg bound_reg --lambda_ratio 0.4 --l1 1e-6 \
--eps_test_L2 0.14117647058 --eps_end_L2 0.14117647058 --joint --lambda_ratio_L2 0.00001 --max --lp --gp --reverse --lbd 2.0"
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
python abcrown.py --config exp_configs/tinyimagenet_l2_2.yaml --complete_verifier skip --share_alphas --onnx_path $MODEL_PATH > out_l2_tiny_ours.txt
python abcrown.py --config exp_configs/tinyimagenet_l1_2.yaml --complete_verifier skip --share_alphas --onnx_path $MODEL_PATH > out_l1_tiny_ours.txt
python abcrown.py --config exp_configs/tinyimagenet_linf_2.yaml --complete_verifier skip --share_alphas --onnx_path $MODEL_PATH > out_linf_tiny_ours.txt

# Step 5: calculate union accuracy
python cal_union.py --dataset tiny --file_path ours

echo "completed."