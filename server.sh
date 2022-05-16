#!/bin/bash

wandb login 2547ade8e5df6e5c103ecf11af5c1453abea046c
python ./lightning_bolts/pl_bolts/models/self_supervised/swav/swav_module_lidc.py \
--save_path ./LIDC_MSD-PreTrain \
--data_dir ./Data/LIDC_MSD \
--model LIDC_MSD_lr-4_ImageNet \
--test LIDC_MSD_lr-4_ImageNet \
--learning_rate 1e-4 \
--load_pretrained_weights  #Diese Argument komplett weglassen wenn ich keine PreTrained weights will **
--pretrained_weights ./Data/ImageNet/swav_800ep_pretrain.pth.tar


# Lokal:
# python /home/wolfda/PycharmProjects/lightning_bolts/pl_bolts/models/self_supervised/swav/swav_module_lidc.py \
# --save_path /home/wolfda/Clinic_Data/Challenge/CT_PreTrain/LIDC/manifest-1600709154662/LIDC-PreTrain \
# --model F \
# --test 0 \
# --data_dir /home/wolfda/Clinic_Data/Challenge/CT_PreTrain/LIDC/manifest-1600709154662/LIDC-2D-jpeg-images

# Lokal ausführen:
# Console:
# 1. chmod +x /home/wolfda/PycharmProjects/lightning_bolts/server.sh (damit das File ausführbar wird [ändert die Permissions]
# 2. /home/wolfda/PycharmProjects/lightning_bolts/server.sh

# Server:
#wandb login 2547ade8e5df6e5c103ecf11af5c1453abea046c
#python ./lightning_bolts/pl_bolts/models/self_supervised/swav/swav_module_lidc.py \
#--save_path ./LIDC_MSD-PreTrain \
#--data_dir ./Data/LIDC_MSD \
#--model LIDC_MSD_lr-4_ImageNet \
#--test LIDC_MSD_lr-4_ImageNet \
#--learning_rate 1e-4 \
#--load_pretrained_weights True \  #Diese Argument komplett weglassen wenn "False" **
#--pretrained_weights ./Data/ImageNet/swav_800ep_pretrain.pth.tar

# ** setzt das default value auf False und es wird True sobald man --load_pretrained_weights benutzt.
