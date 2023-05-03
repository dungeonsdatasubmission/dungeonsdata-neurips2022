from mrunner.helpers.client_helper import get_configuration

# from eval import main
from hackrl.eval import main

if __name__ == "__main__":
    cfg = get_configuration(print_diagnostics=True, with_neptune=False)

    del cfg["experiment_id"]

    main(variant=vars(cfg))
