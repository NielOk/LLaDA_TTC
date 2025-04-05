### This is a script that runs the generate.py script on a running Lambda Cloud instance ###

#!/bin/bash

 cd ../../

 source .env

 cd baseline_inference/generation

# Prompt user for the API key and instance details
read -p "Enter the name of your lambda API key (e.g. niel_lambda_api_key): " user_lambda_api_key_name
USER_LAMBDA_API_KEY=$(eval echo \$$user_lambda_api_key_name)
read -p "Enter the directory location of your private SSH key: " private_ssh_key
read -p "Enter the SSH user (e.g. ubuntu): " remote_ssh_user
read -p "Enter the SSH host/instance address (e.g. 129.146.33.218): " remote_ssh_host

# Copy inference script to the remote instance
GENERATE_SCRIPT_PATH="./generate.py"

read -p "Would you like to copy the inference script to the remote instance? (y/n): " copy_script
if [[ $copy_script == "y" ]]; then
    echo "Copying inference script to remote instance..."
    scp -i "$private_ssh_key" "$GENERATE_SCRIPT_PATH" "$remote_ssh_user@$remote_ssh_host:~/generate.py"
else
    echo "Skipping script copy."
fi

# Install requirements
read -p "Would you like to install the requirements on the remote instance? (y/n): " install_requirements
if [[ $install_requirements == "y" ]]; then
    echo "Installing requirements on remote instance..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "pip install torch numpy transformers jinja2==3.1.0"
else
    echo "Skipping requirements installation."
fi

# Run the infernece script on the remote instance
read -p "Would you like to run the inference script on the remote instance? (y/n): " run_inference
if [[ $run_inference == "y" ]]; then

    # Check if user wants instruct or base model
    read -p "Choose 'base' or 'instruct' model variant." model_variant

    echo "Running inference script on remote instance for model variant: $model_variant..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "nohup python3 ~/generate.py --model_variant $model_variant > generate_output.log 2>&1 &" &
else
    echo "Skipping inference script execution."
fi