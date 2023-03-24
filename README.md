# About

The official implementation of the paper

MULTI-TEMPLATE TRACKER DRIVEN BY CACHE MANAGER ALGORITHM, TOWARDS MULTI-DISTRACTOR SCENARIOS

This repository is based on pytracking and Stark. 
See https://github.com/visionml/pytracking and https://github.com/researchmm/Stark for more help.

# Dataset

GOT-10k http://got-10k.aitestunion.com/ <br /> LaSOT <br /> TrackingNet <br /> MSCOCO

# Environment

```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
lib/train/admin/local.py  # paths for training
lib/test/evaluation/local.py  # paths for testing
```

# Reproduce got10k-only version

## Train
Download STARK-ST50_got10k_only checkpoint from https://github.com/researchmm/Stark/blob/main/MODEL_ZOO.md
```
mv %checkpoint% ./checkpoints/train/stark_st2/baseline_got10k_only
# Training step 1
python ./tracking/train.py --script cachet_4_l6 --config baseline_got10k_only --save_dir . --mode multiple --nproc_per_node N --script_prv stark_st2 --config_prv baseline_got10k_only
# Training step 2
python ./tracking/train.py --script cachetcls_4_l6 --config baseline_got10k_only --save_dir . --mode multiple --nproc_per_node N --script_prv cachet_4_l6 --config_prv baseline_got10k_only
```

## Debug

```
python ./tracking/test.py cache_tracker baseline_got10k_only --dataset DATASET --sequence SEQUENCE --debug 1
```

## Evaluate
```
python ./tracking/test.py cache_tracker baseline_got10k_only --dataset got10k_test --debug 0 --threads THREADS --num_gpu NUM_GPUS
```

## Pack and submit results
see https://github.com/visionml/pytracking/blob/master/pytracking/util_scripts/pack_got10k_results.py

