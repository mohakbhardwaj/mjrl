

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



conda create -n pmpc python=3.7
conda activate pmpc
git clone https://github.com/mohakbhardwaj/mjrl.git
cd mjrl
git checkout v2
# . install_mujoco.sh  # install mujoco, if missing
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
pip install -r requirements.txt
pip insatll -e  .
