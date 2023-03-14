deepspeed --num_gpus=8 finetune.py --config_path ./configs/base_configs/bpt.yaml \
--ds_config_path ./configs/ds_configs/ds_config_bpt.json \
--deepspeed ./configs/ds_configs/ds_config_bpt.json