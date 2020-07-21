
TRAIN_SET="../data/msmd/msmd_train"
VAL_SET="../data/msmd/msmd_valid"
LOG_ROOT="ismir_runs"
DUMP_ROOT="ismir_params"

CONFIG="configs/msmd.yaml"
CONFIG_AUG="configs/msmd_aug.yaml"


# CB Encoder without Tempo Augmentation
if [ $1 -eq 0 ]
then
    CUDA_VISIBLE_DEVICES=0 python train_rnn.py --film_layers 2 3 4 5 6 7 8 --log_root $LOG_ROOT --dump_root $DUMP_ROOT --train_set $TRAIN_SET --val_set $VAL_SET --use_lstm --augment --config $CONFIG --audio_encoder CBEncoder --tag CB_noTA
fi

## CB Encoder with Tempo Augmentation
if [ $1 -eq 1 ]
then
    CUDA_VISIBLE_DEVICES=1 python train_rnn.py --film_layers 2 3 4 5 6 7 8 --log_root $LOG_ROOT --dump_root $DUMP_ROOT --train_set $TRAIN_SET --val_set $VAL_SET --use_lstm --augment --config $CONFIG_AUG --audio_encoder CBEncoder --tag CB_TA
fi

# FB Encoder without Tempo Augmentation
if [ $1 -eq 2 ]
then
  CUDA_VISIBLE_DEVICES=0 python train_rnn.py --film_layers 2 3 4 5 6 7 8 --log_root $LOG_ROOT --dump_root $DUMP_ROOT --train_set $TRAIN_SET --val_set $VAL_SET --use_lstm --augment --config $CONFIG  --audio_encoder $ENCODER --tag FB_noTA
fi

## FB Encoder with Tempo Augmentation
if [ $1 -eq 3 ]
then
  CUDA_VISIBLE_DEVICES=1 python train_rnn.py --film_layers 2 3 4 5 6 7 8 --log_root $LOG_ROOT --dump_root $DUMP_ROOT --train_set $TRAIN_SET --val_set $VAL_SET --use_lstm --augment --config $CONFIG_AUG --audio_encoder $ENCODER --tag FB_TA
fi

# NTC without Tempo Augmentation
if [ $1 -eq 4 ]
then
    CUDA_VISIBLE_DEVICES=0 python train_rnn.py --film_layers 2 3 4 5 6 7 8 --log_root $LOG_ROOT --dump_root $DUMP_ROOT --train_set $TRAIN_SET --val_set $VAL_SET --augment --config $CONFIG  --audio_encoder CBEncoder --tag NTC_noTA
fi

## NTC with Tempo Augmentation
if [ $1 -eq 5 ]
then
    CUDA_VISIBLE_DEVICES=1 python train_rnn.py --film_layers 2 3 4 5 6 7 8 --log_root $LOG_ROOT --dump_root $DUMP_ROOT --train_set $TRAIN_SET --val_set $VAL_SET --augment --config $CONFIG_AUG  --audio_encoder CBEncoder --tag NTC_TA
fi



