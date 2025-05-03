### This is a script that runs the generate.py script on a running Lambda Cloud instance ###

#!/bin/bash

cd ../../

source .env

cd meta_cognition/decoding_policy_based_generation/

# Prompt user for the API key and instance details
read -p "Enter the name of your lambda API key (e.g. niel_lambda_api_key): " user_lambda_api_key_name
USER_LAMBDA_API_KEY=$(eval echo \$$user_lambda_api_key_name)
read -p "Enter the directory location of your private SSH key: " private_ssh_key
read -p "Enter the SSH user (e.g. ubuntu): " remote_ssh_user
read -p "Enter the SSH host/instance address (e.g. 129.146.33.218): " remote_ssh_host

# Copy inference scripts to the remote instance

cd ..
pwd
GENERATE_SCRIPT_NAME="test_generate_with_decoding_policy.py"
DECODING_POLICY_NAME="decoding_policy_state.py"
UTILS_NAME="policy_based_decoding_utils.py"
echo "Copying inference script to remote instance..."

read -p "Would you like to copy the inference script to the remote instance? (y/n): " copy_script
if [[ $copy_script == "y" ]]; then
    echo "Copying inference script to remote instance..."
    scp -i "$private_ssh_key" "./decoding_policy_based_generation/$GENERATE_SCRIPT_NAME" "$remote_ssh_user@$remote_ssh_host:~/$GENERATE_SCRIPT_NAME"
    scp -i "$private_ssh_key" "./MCTS/$DECODING_POLICY_NAME" "$remote_ssh_user@$remote_ssh_host:~/$DECODING_POLICY_NAME"
    scp -i "$private_ssh_key" "./MCTS/$UTILS_NAME" "$remote_ssh_user@$remote_ssh_host:~/$UTILS_NAME"
else
    echo "Skipping script copy."
fi

cd decoding_policy_based_generation/

# Install requirements
read -p "Would you like to install the requirements on the remote instance? (y/n): " install_requirements
if [[ $install_requirements == "y" ]]; then
    echo "Installing requirements on remote instance..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "pip install torch numpy transformers accelerate jinja2==3.1.0"
else
    echo "Skipping requirements installation."
fi

# Run the inference script on the remote instance
read -p "Would you like to run the inference script on the remote instance? (y/n): " run_inference
if [[ $run_inference == "y" ]]; then

    # Check if user wants instruct or base model
    read -p "Choose 'base' or 'instruct' model variant: " model_variant

    echo "Running inference script on remote instance for model variant: $model_variant..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "nohup python3 ~/$GENERATE_SCRIPT_NAME --model_variant $model_variant > ${model_variant}_generate_output.log 2>&1 &" &
else
    echo "Skipping inference script execution."
fi