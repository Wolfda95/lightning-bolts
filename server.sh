#!/bin/bash

wandb login 2547ade8e5df6e5c103ecf11af5c1453abea046c
python ./lightning_bolts/pl_bolts/models/self_supervised/swav/swav_module_lidc.py \
--save_path ./LIDC_MSD-PreTrain \
--data_dir ./Data/LIDC_MSD \
--model LIDC_MSD_lr-4 \
--test LIDC_MSD_lr-4 \
--learning_rate 1e-4 \


# Lokal:
# python /home/wolfda/PycharmProjects/lightning_bolts/pl_bolts/models/self_supervised/swav/swav_module_lidc.py \
# --save_path /home/wolfda/Clinic_Data/Challenge/CT_PreTrain/LIDC/manifest-1600709154662/LIDC-PreTrain \
# --model F \
# --test 0 \
# --data_dir /home/wolfda/Clinic_Data/Challenge/CT_PreTrain/LIDC/manifest-1600709154662/LIDC-2D-jpeg-images

# Docker:
#wandb login 2547ade8e5df6e5c103ecf11af5c1453abea046c
#python ./lightning_bolts/pl_bolts/models/self_supervised/swav/swav_module_lidc.py \
#--save_path ./LIDC-PreTrain \
#--model F \
#--test 0 \
#--data_dir ./LIDC-2D-jpeg-images


# Lokal ausführen:
# Console:
# 1. chmod +x /home/wolfda/PycharmProjects/lightning_bolts/server.sh (damit das File ausführbar wird [ändert die Permissions]
# 2. /home/wolfda/PycharmProjects/lightning_bolts/server.sh
