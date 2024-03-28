from mrunner.helpers.client_helper import get_configuration

from hackrl.eval import main as main_eval
from hackrl.rollout import main as main_rollout

MAIN_DICT = {"eval": main_eval, "rollout": main_rollout}


if __name__ == "__main__":
    config = get_configuration(print_diagnostics=True, with_neptune=False)

    del config["experiment_id"]
    run_kind = config.pop("run_kind")

    main = MAIN_DICT[run_kind]
    main(variant=vars(config))
