import subprocess

from mrunner.helpers.client_helper import get_configuration


if __name__ == "__main__":
    cfg = get_configuration(print_diagnostics=True, with_neptune=False)

    del cfg["experiment_id"]

    # Start the moolib.broker module in a new process
    broker_process = subprocess.Popen(['python', '-m', 'moolib.broker'])

    key_pairs = [f"{key}={value}" for key, value in cfg.items()]
    cmd = ['python', '-m', 'experiment_code.hackrl.experiment'] + key_pairs
    subprocess.run(cmd) 

    # When you're done, terminate the broker process
    broker_process.terminate()
