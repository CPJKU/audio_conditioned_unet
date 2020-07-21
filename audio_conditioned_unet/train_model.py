
import argparse
import copy
import json
import os
import random
import torch
import sys


import numpy as np
import multiprocessing as mp

from audio_conditioned_unet.dataset import iterate_dataset, load_dataset, NonSequentialDatasetWrapper
from audio_conditioned_unet.network import ConditionalUNet
from audio_conditioned_unet.utils import load_game_config

from time import gmtime, strftime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Script for ISMIR 2020')
    parser.add_argument('--film_layers', nargs='+', help='list of block indices where conditioning should be applied', type=int)
    parser.add_argument('--augment', help='activate data augmentation', default=False, action='store_true')
    parser.add_argument('--tempo_augment', help='activate tempo augmentation', default=False, action='store_true')
    parser.add_argument('--train_set', help='path to train dataset.', type=str)
    parser.add_argument('--val_set', help='path to validation dataset.', type=str)
    parser.add_argument('--batch_size', help='batch size.', type=int, default=4)
    parser.add_argument('--seq_len', help='sequence length for training', type=int, default=16)
    parser.add_argument('--log_root', help='path to log directory', type=str, default="runs")
    parser.add_argument('--dump_root', help='name for the stored network', type=str, default="params")
    parser.add_argument('--tag', help='additional tag', type=str, default="")
    parser.add_argument('--n_encoder_layers', '--enc', help='number of encoding layers.', type=int, default=4)
    parser.add_argument('--n_filters_start', '--filters', help='number of initial filters.', type=int, default=8)
    parser.add_argument('--rnn_size', help='number of rnn units.', type=int, default=128)
    parser.add_argument('--spec_enc', help='number of hidden units for the dense layer before the rnn', type=int, default=32)
    parser.add_argument('--rnn_layer', help='number of rnn layer.', type=int, default=1)
    parser.add_argument('--learning_rate', "--lr", help='learning rate.', type=float, default=1e-4)
    parser.add_argument('--scale_factor', help='sheet image scale factor.', type=int, default=3)
    parser.add_argument('--weight_decay', help='weight decay value.', type=float, default=1e-5)
    parser.add_argument('--param_path', help='load network weights', type=str, default=None)
    parser.add_argument('--no_save', help='do not save parameters', default=False, action='store_true')
    parser.add_argument('--use_lstm', help='if set use LSTM otherwise no long-term temporal context is used', default=False, action='store_true')
    parser.add_argument('--all_tempi', help='use all tempi for augmentation', default=False, action='store_true')
    parser.add_argument('--config', help='path to config.', type=str)
    parser.add_argument('--patience', help='patience before decreasing the learning rate.', type=int, default=5)
    parser.add_argument('--seed', help='random seed.', type=int, default=4711)
    parser.add_argument('--audio_encoder', help='audio encoder', type=str, default="CBEncoder")
    parser.add_argument('--clip_grads', help='gradient clipping value', type=float, default=None)

    args = parser.parse_args()

    # set random seed and variables for reproducibility https://pytorch.org/docs/stable/notes/randomness.html
    # unfortunately there is still some source of non-determinism possibly e.g. due to nn.Upsample
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # makes training/evaluation very slow
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # apparently the code gets stuck without this line when computing the spectrograms
    mp.set_start_method('spawn', force=True)

    config = load_game_config(args.config)
    train_path = args.train_set
    val_path = args.val_set

    time_stamp = strftime("%Y%m%d_%H%M%S", gmtime()) + "_{}".format(args.tag)

    if not os.path.exists(args.log_root):
        os.mkdir(args.log_root)

    if not os.path.exists(args.dump_root):
        os.mkdir(args.dump_root)

    dump_path = os.path.join(args.dump_root, time_stamp)

    if not os.path.exists(dump_path):
        os.mkdir(dump_path)

    train_parameters = dict(
        num_epochs=100,
        batch_size=args.batch_size,
        max_patience=args.patience*2,   # max patience before stopping training is twice the patience used to reduce learn rate
        lr=args.learning_rate,
        dump_path=dump_path,
        augment=args.augment,
        tempo_augment=args.tempo_augment,
        seq_len=args.seq_len
    )

    log_dir = os.path.join(args.log_root, time_stamp)
    log_writer = SummaryWriter(log_dir=log_dir)

    text = ""
    arguments = np.sort([arg for arg in vars(args)])
    for arg in arguments:
        text += "**{}:** {}<br>".format(arg, getattr(args, arg))

    for key in train_parameters.keys():
        text += "**{}:** {}<br>".format(key, train_parameters[key])

    log_writer.add_text("run_config", text)
    log_writer.add_text("cmd", " ".join(sys.argv))

    net_config = {'film_layers': args.film_layers,
                  'n_encoder_layers': args.n_encoder_layers,
                  'n_filters_start': args.n_filters_start,
                  'rnn_size': args.rnn_size,
                  'rnn_layer': args.rnn_layer,
                  'use_lstm': args.use_lstm,
                  'audio_encoder': args.audio_encoder,
                  'spec_enc': args.spec_enc}

    # store the network configuration
    with open(os.path.join(dump_path, 'net_config.json'), "w") as f:
        json.dump(net_config, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = ConditionalUNet(net_config)

    if args.param_path is not None:
        print('Loading model from {}'.format(args.param_path))
        network.load_state_dict(torch.load(args.param_path, map_location=lambda storage, location: storage))

    # no augmentation on the validation set
    val_config = copy.deepcopy(config)
    val_config['tempo_factors'] = [1000]

    n_frames = network.perf_encoder.n_input_frames
    train_dataset = load_dataset(train_path, config, n_frames=n_frames, augment=train_parameters['augment'],
                                 scale_factor=args.scale_factor, all_tempi=args.all_tempi)
    val_dataset = load_dataset(val_path, val_config, n_frames=n_frames, augment=False, scale_factor=args.scale_factor)

    specs = [train_dataset.performances[elem][1000]['spec'] for elem in train_dataset.performances]
    means = np.mean(np.concatenate(specs, axis=-1), axis=1)
    stds = np.std(np.concatenate(specs, axis=-1), axis=1)
    network.perf_encoder.set_stats(means, stds)

    if not args.use_lstm:
        # dataset wrapper to use same script for sequential and non sequential case
        train_dataset = NonSequentialDatasetWrapper(train_dataset)
        val_dataset = NonSequentialDatasetWrapper(val_dataset)

    print(f"Putting model to {device}")
    network.to(device)

    optim = torch.optim.Adam(network.parameters(), lr=train_parameters['lr'], weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optim, mode="min", patience=args.patience, factor=0.5, verbose=True)

    patience = train_parameters['max_patience']
    min_loss = np.infty

    batch_size = train_parameters['batch_size']

    for epoch in range(train_parameters['num_epochs']):

        tr_stats = iterate_dataset(network, optim, train_dataset, batch_size, seq_len=train_parameters['seq_len'],
                                train=True, device=device, threshold=0.5, clip_grads=args.clip_grads)
        tr_loss, tr_prec, tr_rec = tr_stats['loss'], tr_stats['precision'], tr_stats['recall']

        val_stats = iterate_dataset(network, None, val_dataset, batch_size=batch_size, seq_len=train_parameters['seq_len'],
                                    train=False, device=device, threshold=0.5)
        val_loss, val_prec, val_rec = val_stats['loss'], val_stats['precision'], val_stats['recall']

        scheduler.step(val_loss)

        if val_loss < min_loss:
            min_loss = val_loss
            color ='\033[92m'
            patience = train_parameters['max_patience']

            if not args.no_save:
                print("Store best model...")
                torch.save(network.state_dict(), os.path.join(train_parameters['dump_path'], "best_model.pt"))
        else:
            color = '\033[91m'
            patience -= 1

        # store latest model
        if not args.no_save:
            torch.save(network.state_dict(), os.path.join(train_parameters['dump_path'], "latest_model.pt".format(epoch)))

        tr_f1 = 2*(tr_prec*tr_rec)/(tr_prec + tr_rec) if tr_prec > 0 and tr_prec > 0 else 0
        val_f1 = 2*(val_prec*val_rec)/(val_prec + val_rec) if val_prec > 0 and val_prec > 0 else 0

        log_writer.add_scalar('training/loss', tr_loss, epoch)
        log_writer.add_scalar('training/precision', tr_prec, epoch)
        log_writer.add_scalar('training/recall', tr_rec, epoch)
        log_writer.add_scalar('training/f1', tr_f1, epoch)

        log_writer.add_scalar('training/lr', optim.param_groups[0]['lr'], epoch)

        log_writer.add_scalar('validation/loss', val_loss, epoch)
        log_writer.add_scalar('validation/precision', val_prec, epoch)
        log_writer.add_scalar('validation/recall', val_rec, epoch)
        log_writer.add_scalar('validation/f1', val_f1, epoch)

        print("\n{}Epoch {} | Train Loss: {}, Precision: {}, Recall: {}\033[0m".format(color, epoch, tr_loss, tr_prec, tr_rec))
        print("{}Epoch {} | Val Loss: {}, Precision: {}, Recall: {}\033[0m".format(color, epoch, val_loss, val_prec, val_rec))




