#!/bin/bash


python -m utils.stitch_evals --json utils/stitch_configs/stitch_monk_appo_ks_T.json
python -m utils.stitch_evals --json utils/stitch_configs/stitch_monk_appo_ks.json
python -m utils.stitch_evals --json utils/stitch_configs/stitch_monk_appo_bc_T.json
python -m utils.stitch_evals --json utils/stitch_configs/stitch_monk_appo_bc.json
python -m utils.stitch_evals --json utils/stitch_configs/stitch_monk_appo_T.json 
python -m utils.stitch_evals --json utils/stitch_configs/stitch_monk_appo.json
