
# Clone the repo
# git clone -b v2 https://github.com/mohakbhardwaj/mjrl.git
# cd mjrl
# . install.sh

conda create -n pmpc python=3.7 -y
conda activate pmpc
# . install_mujoco.sh  # install mujoco, if missing
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
pip install -r requirements.txt
pip install -e  .



# old one
# git clone https://github.com/mohakbhardwaj/mjrl.git
# cd mjrl
# git checkout v2
# . install_mujoco.sh
# conda env create -f setup/env.yml
# conda activate mjrl-env

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
# pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
# pip install -e .
# python prep_d4rl_dataset.py  --env_name hopper-medium-v2
