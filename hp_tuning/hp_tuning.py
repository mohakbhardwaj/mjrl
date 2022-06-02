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


def train(config, seed, datapath, **job_data):
    """ config: path (relative to run_mpc.py) to the config file. A default job_data dict is loaded from this file.
        job_data: a dict that contains the hyperparameters to overwrite the default job_data.
    """
    from rl_nexus.mjrl.projects.pessimistic_mpc.run_mpc import train
    from rl_nexus.mjrl.mjrl.utils.utils import parse_and_update_dict
    import os, sys

    # Load config
    base_path = os.path.join(os.getcwd(),'mjrl','projects','pessimistic_mpc')
    path = os.path.join(base_path, config)
    sys.path.append(base_path)
    with open(path, 'r') as f:
        job_data0 = eval(f.read())

    job_data = parse_and_update_dict(job_data0, job_data, token='-')

    return train(job_data=job_data,
                 output='../results/exp_data',
                 datapath=datapath,
                 seed=seed)

def run(hp_tuning_mode='grid',
        n_seeds_per_hp=3):


    hps_dict = {
        'config':['configs/d4rl_hopper_medium.txt'],
        'mpc_params-horizon':[10, 20],
    }


    config = dict(
        seed='randint',
        modelpath='$datastore/pessimistic_mpc/cached_models'
        readonly=True,
    )

    xt_setup = {
      'activate':None,
      'other-cmds':[
          "cd rl_nexus",
          "ls",
          "git clone -b v2 https://github.com/mohakbhardwaj/mjrl.git",
          "cd mjrl",
          ". install_mujoco.sh",   # install mujoco210
          "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia",
          "pip install -r requirements.txt",
          "pip install -e  . ",  # install mjrl
          "cd ../../",   # dilbert directory
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
                  # remote_run=False,
                  )

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    # parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--hp_tuning_mode', type=str, default='grid')
    parser.add_argument('--n_seeds_per_hp', type=int, default=3)

    run(**vars(parser.parse_args()))
