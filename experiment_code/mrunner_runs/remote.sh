#!/bin/bash

conda activate dungeons

ssh-add
export PYTHONPATH=.

# mrunner --config ~/.mrunner.yaml --context athena_nethack run mrunner_exps/APPO/@-APPO.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack run mrunner_exps/APPO/monk-APPO.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack run mrunner_exps/BC/@-AA-BC.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack run mrunner_exps/BC/monk-AA-BC.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack run mrunner_exps/BC/@-NAO-BC.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack run mrunner_exps/APPO-BC/@-APPO-AA-BC.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack run mrunner_exps/APPO-BC/monk-APPO-AA-BC.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack run mrunner_exps/APPO-BC/@-APPO-NAO-BC.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack run mrunner_exps/APPO-KS/@-APPO-AA-KS.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack run mrunner_exps/APPO-KS/monk-APPO-AA-KS.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack run mrunner_exps/APPO-KS/@-APPO-NAO-KS.py


# mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/paper_baselines/2023_11_05_monk-APPO.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/paper_baselines/2023_11_05_monk-APPO-T.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/paper_baselines/2023_11_05_monk-APPO-AA-KS.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/paper_baselines/2023_11_05_monk-APPO-AA-KS-T.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/paper_baselines/2023_11_05_monk-APPO-AA-KL.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/paper_baselines/2023_11_05_monk-APPO-AA-KL-T.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/paper_baselines/2023_11_05_monk-APPO-AA-BC.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/paper_baselines/2023_11_05_monk-APPO-AA-BC-T.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_1gpu run mrunner_exps/paper_baselines/2023_11_05_monk-AA-BC.py