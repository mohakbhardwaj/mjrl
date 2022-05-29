# To set up a job, the user needs to specify `method` and `hp_dict` and then
# import `submit_xt_job`. Optionally, the user can provide additional arguments
# `submit_xt_job` takes. To run the job, go to the `rl_nexus` directory and then
# execute this python file.

import numpy as np
from rl_nexus.hp_tuning_tools import submit_xt_job
import os

# hp compute
max_total_runs=2000
n_concurrent_runs_per_node=1
max_n_nodes=1
compute_target='azb-gpu'
docker_image = 'mujoco'
azure_service='dilbertbatch' #'dilbertbatch' #'rdlbatches' # 'rdlbatches' # dilbertbatch'

code_paths = os.path.dirname(__file__)  # This file will be uploaded as rl_nexus/hp_tuning/hp_tuning.py
method = 'rl_nexus.hp_tuning.hp_tuning.train' # so we call the method below as left.


# $datastore/pmpc/...  # TODO
def train(config, seed, **job_data):
    """ config: path (relative to run_mpc.py) to the config file. A default job_data dict is loaded from this file.
        job_data: a dict that contains the hyperparameters to overwrite the default job_data.
    """
    from rl_nexus.mjrl.projects.pessimistic_mpc.run_mpc import train
    # from rl_nexus.mjrl.projects.pessimistic_mpc import train
    from rl_nexus.mjrl.mjrl.utils.utils import parse_and_update_dict
    import os, sys

    base_path = os.path.join(os.getcwd(),'mjrl','projects','pessimistic_mpc')
    path = os.path.join(base_path, config)
    sys.path.append(base_path)
    # Load config
    with open(path, 'r') as f:
        job_data0 = eval(f.read())

    job_data = parse_and_update_dict(job_data0, job_data, token='-')


    return train(job_data=job_data,
                 output='../results/exp_data',
                 seed=seed)

def run(hp_tuning_mode='grid',
        n_seeds_per_hp=3):


    hps_dict = {
        'config':['configs/d4rl_hopper_medium.txt'],
        'mpc_params-horizon':[10, 20],
    }


    config = dict(
        seed='randint',
    )

    xt_setup = {  # mujoco200
      'activate':None,
      'other-cmds':[
          # "sudo apt-get install -y patchelf",
          # "export MUJOCO_PY_MJKEY_PATH=/opt/mjkey.txt",
          # "export MUJOCO_PY_MUJOCO_PATH=/opt/mujoco200_linux",
          # "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mujoco200_linux/bin",
          # "cd /root",
          # "ln -s /opt .mujoco",  # for d4rl # error: Header file '/root/.mujoco/mujoco200_linux/include/mjdata.h' does not exist./
          # "cp -r .mujoco/mujoco200_linux  .mujoco/mujoco210",   # error: Header file '/root/.mujoco/mujoco210/include/mjdata.h' does not exist.
          # "cd -",  # dilbert directory
          "cd rl_nexus",
          "echo DEBUG: RL_NEUXS DIR",
          "ls",
          # install mjrl
          "git clone -b v2 https://github.com/mohakbhardwaj/mjrl.git",
          "cd mjrl",
          "echo DEBUG: INSTALL MUJOCO",
          ". install_mujoco.sh",   # install mujoco, if missing
          "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia",
          "pip install -r requirements.txt",
          "pip install -e  . ",  # install mjrl
          "echo DEBUG: FINISHED INSTALLING MJRL",
          "cd ../../",   # dilbert directory
          "echo DEBUG: RL_NEUXS DIR AFTER INSTALL",
          "ls rl_nexus",
          ],
      'conda-packages': [],
      'pip-packages': [],
      'python-path': ["../"]
    }

    submit_xt_job(method,
                  hps_dict,
                  config=config,
                  n_concurrent_runs_per_node=n_concurrent_runs_per_node,
                  xt_setup=xt_setup,
                  hp_tuning_mode=hp_tuning_mode,
                  n_seeds_per_hp=n_seeds_per_hp,
                  max_total_runs=max_total_runs,
                  max_n_nodes=max_n_nodes,
                  azure_service=azure_service,
                  code_paths=code_paths,
                  docker_image=docker_image,
                  compute_target=compute_target,
                  remote_run=False,
                  )

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    # parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--hp_tuning_mode', type=str, default='grid')
    parser.add_argument('--n_seeds_per_hp', type=int, default=3)

    run(**vars(parser.parse_args()))
