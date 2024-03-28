import subprocess

import torch

from mrunner.helpers.client_helper import get_configuration


if __name__ == "__main__":
    cfg = get_configuration(print_diagnostics=True, with_neptune=False)

    del cfg["experiment_id"]

    # Start the moolib.broker module in a new process
    broker_process = subprocess.Popen(['python', '-m', 'moolib.broker'])

    key_pairs = [f"{key}={value}" for key, value in cfg.items()]
    cmd = ['python', '-m', 'hackrl.experiment'] + key_pairs

    device_count = torch.cuda.device_count()
    if device_count > 1:
        if "device" not in cfg or cfg["device"] != "cpu":
            # Adding more peers to this experiment, starting more processes with the
            # same `project` and `group` settings, using a different setting for `device`           
            for i in range(1, device_count):
                subprocess.Popen(cmd + [f"device=cuda:{i}"])

    # default device is cuda:0
    subprocess.run(cmd) 

    # When you're done, terminate the broker process
    broker_process.terminate()
