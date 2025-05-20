### This is a script that runs the block_schedule_and_extra_step_proportion.py script on a running Lambda Cloud instance ###

#!/bin/bash

cd ../../

source .env

cd meta_cognition/grpo_embedded_mcts/

# Prompt user for the API key and instance details
read -p "Enter the name of your lambda API key (e.g. niel_lambda_api_key): " user_lambda_api_key_name
USER_LAMBDA_API_KEY=$(eval echo \$$user_lambda_api_key_name)
read -p "Enter the directory location of your private SSH key: " private_ssh_key
read -p "Enter the SSH user (e.g. ubuntu): " remote_ssh_user
read -p "Enter the SSH host/instance address (e.g. 129.146.33.218): " remote_ssh_host

# Copy inference scripts to the remote instance
GRPO_EMBEDDED_MCTS_SCRIPT="./grpo_embedded_monte_carlo_tree_search.py"
DECODING_POLICY_NAME="./decoding_policy_state.py"
DECODING_UTILS_NAME="./policy_based_decoding_utils.py"
MCTS_NODE_NAME="./mcts_node.py"
MCTS_UTILS_NAME="./grpo_embedded_mcts_utils.py"
echo "Copying inference script to remote instance..."

read -p "Would you like to copy the inference scripts to the remote instance? (y/n): " copy_script
if [[ $copy_script == "y" ]]; then
    echo "Copying inference script to remote instance..."
    scp -i "$private_ssh_key" "$GRPO_EMBEDDED_MCTS_SCRIPT" "$remote_ssh_user@$remote_ssh_host:~/$GRPO_EMBEDDED_MCTS_SCRIPT"
    scp -i "$private_ssh_key" "$DECODING_POLICY_NAME" "$remote_ssh_user@$remote_ssh_host:~/$DECODING_POLICY_NAME"
    scp -i "$private_ssh_key" "$DECODING_UTILS_NAME" "$remote_ssh_user@$remote_ssh_host:~/$DECODING_UTILS_NAME"
    scp -i "$private_ssh_key" "$MCTS_NODE_NAME" "$remote_ssh_user@$remote_ssh_host:~/$MCTS_NODE_NAME"
    scp -i "$private_ssh_key" "$MCTS_UTILS_NAME" "$remote_ssh_user@$remote_ssh_host:~/$MCTS_UTILS_NAME"

    # Copy .env file to remote instance
    cd ../../
    scp -i "$private_ssh_key" ".env" "$remote_ssh_user@$remote_ssh_host:~/.env"
    cd meta_cognition/grpo_embedded_mcts/
else
    echo "Skipping script copy."
fi

# Install requirements
read -p "Would you like to install the requirements on the remote instance? (y/n): " install_requirements
if [[ $install_requirements == "y" ]]; then
    echo "Installing requirements on remote instance..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "pip install openai torch numpy transformers accelerate jinja2==3.1.0 datasets python-dotenv huggingface_hub"
else
    echo "Skipping requirements installation."
fi

# Run the GRPO-embedded MCTS script on the remote instance
TRAINED_TREES_DIR="./trained_trees"
GRPO_EMBEDDED_MCTS_PRETRAINED_TREE_JSON_NAME="pre_training_grpo_embedded_mcts_tree_snapshot.json"
GRPO_EMBEDDED_MCTS_PRETRAINED_TREE_METADATA_JSON_NAME="pre_training_grpo_embedded_mcts_metadata.json"
read -p "Would you like to run the GRPO-embedded MCTS script on the remote instance? (y/n): " run_mode
if [[ $run_mode == "y" ]]; then # pre-training from scratch

    echo "Running GRPO-embedded MCTS script for pre-training from scratch on remote instance for model variant: Instruct..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "nohup python3 -u ~/$GRPO_EMBEDDED_MCTS_SCRIPT > pre_train_from_scratch_instruct_grpo_embedded_mcts_output.log 2>&1 &" &
else
    echo "Skipping GRPO-embedded MCTS script execution."
fi