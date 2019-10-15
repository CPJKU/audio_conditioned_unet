
import json
import os
import torch
import sys

import numpy as np

from audio_conditioned_unet.dataset import iterate_dataset, load_dataset
from audio_conditioned_unet.lr_scheduler import CustomReduceLROnPlateau
from audio_conditioned_unet.network import UNetModular

from tensorboardX import SummaryWriter
from time import gmtime, strftime
from torch.utils.data import DataLoader

SF_PATH = 'sound_fonts/grand-piano-YDP-20160804.sf2'


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Train Script for WoRMS Workshop')

    for i in range(1, 10):
        parser.add_argument('--film{}'.format(i), help='activate film layer {}'.format(i),
                            default=False, action='store_true')

    parser.add_argument('--augment', help='activate data augmentation', default=False, action='store_true')
    parser.add_argument('--tempo_augment', help='activate tempo augmentation', default=False, action='store_true')
    parser.add_argument('--train_set', help='path to train dataset.', type=str)
    parser.add_argument('--val_set', help='path to validation dataset.', type=str)
    parser.add_argument('--batch_size', help='batch size.', type=int, default=32)
    parser.add_argument('--log_root', help='path to log directory', type=str, default="runs")
    parser.add_argument('--dump_root', help='path to the dump directory where the parameters will be stored',
                        type=str, default="params")
    parser.add_argument('--tag', help='additional tag to identify your experiment', type=str, default="")
    parser.add_argument('--activation', '--act', help='activation_function [relu|elu]', type=str, default="relu")
    parser.add_argument('--n_encoder_layers', '--enc', help='number of encoding layers.', type=int, default=4)
    parser.add_argument('--n_filters_start', '--filters', help='number of initial filters.', type=int, default=8)
    parser.add_argument('--spec_out', help='number of output features for the spectrogram.', type=int, default=128)
    parser.add_argument('--learning_rate', "--lr", help='learning rate.', type=float, default=1e-3)
    parser.add_argument('--dropout', help='activate dropout', default=False, action='store_true')
    parser.add_argument('--frame_size', help='spectrogram frame size.', type=int, default=40)
    parser.add_argument('--only_onsets', help='only learn on onset frames', default=False, action='store_true')
    parser.add_argument('--scale_factor', help='sheet image scale factor.', type=int, default=3)

    args = parser.parse_args()

    train_path = args.train_set
    val_path = args.val_set

    time_stamp = strftime("%Y%m%d_%H%M%S", gmtime()) +"_{}".format(args.tag)

    if not os.path.exists(args.log_root):
        os.mkdir(args.log_root)

    if not os.path.exists(args.dump_root):
        os.mkdir(args.dump_root)

    dump_path = os.path.join(args.dump_root, time_stamp)

    if not os.path.exists(dump_path):
        os.mkdir(dump_path)

    train_parameters = dict(
        num_epochs=1000,
        batch_size=args.batch_size,
        max_reductions=5,
        lr=args.learning_rate,
        dump_path=dump_path,
        augment=args.augment,
        tempo_augment=args.tempo_augment
    )

    log_dir = os.path.join(args.log_root, time_stamp)
    log_writer = SummaryWriter(log_dir=log_dir)

    text = ""
    arguments = np.sort([arg for arg in vars(args)])
    for arg in arguments:
        text += "**{}:** {}<br>".format(arg, getattr(args, arg))
        # log_writer.add_text(arg, "{}".format(getattr(args, arg)))

    for key in train_parameters.keys():
        text += "**{}:** {}<br>".format(key, train_parameters[key])

    log_writer.add_text("run_config", text)
    log_writer.add_text("cmd", " ".join(sys.argv))

    net_config = {}
    for i in range(1, args.n_encoder_layers*2+1):
        layer = 'film{}'.format(i)
        net_config[layer] = getattr(args, layer)

    net_config['n_encoder_layers'] = args.n_encoder_layers
    net_config['n_filters_start'] = args.n_filters_start
    net_config['activation'] = args.activation
    net_config['spec_out'] = args.spec_out
    net_config['dropout'] = args.dropout

    # store the network configuration
    with open(os.path.join(dump_path, 'net_config.json'), "w") as f:
        json.dump(net_config, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = load_dataset(train_path, sf_path=SF_PATH, augment=train_parameters['augment'],
                                 tempo_augment=train_parameters['tempo_augment'], frame_size=args.frame_size,
                                 only_onsets=args.only_onsets, scale_factor=args.scale_factor)
    val_dataset = load_dataset(val_path, sf_path=SF_PATH, augment=False, tempo_augment=False, frame_size=args.frame_size,
                               only_onsets=args.only_onsets, scale_factor=args.scale_factor)


    network = UNetModular(net_config)

    train_loader = DataLoader(train_dataset, batch_size=train_parameters['batch_size'], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=train_parameters['batch_size'], shuffle=False, num_workers=8)

    print("Putting model to %s ..." % device)
    network.to(device)

    optim = torch.optim.Adam(network.parameters(), lr=train_parameters['lr'], weight_decay=5e-5)

    scheduler = CustomReduceLROnPlateau(optim, mode="min", patience=2, factor=0.5, verbose=True)

    min_loss = np.infty

    for epoch in range(train_parameters['num_epochs']):

        tr_loss, (tr_prec, tr_rec) = iterate_dataset(network, optim, train_loader, epoch, num_epochs=train_parameters['num_epochs'], train=True, device=device, threshold=0.5)
        val_loss, (val_prec, val_rec) = iterate_dataset(network, None, val_loader, epoch, num_epochs=train_parameters['num_epochs'], train=False, device=device, threshold=0.5)
        scheduler.step(val_loss)

        if val_loss < min_loss:
            min_loss = val_loss
            color ='\033[92m'

            # store best model so far
            torch.save(network.state_dict(), os.path.join(train_parameters['dump_path'], "best_model.pt"))
        else:
            color = '\033[91m'

        # store latest model
        torch.save(network.state_dict(), os.path.join(train_parameters['dump_path'], "latest_model.pt".format(epoch)))

        log_writer.add_scalar('training/loss', tr_loss, epoch)
        log_writer.add_scalar('training/precision', tr_prec, epoch)
        log_writer.add_scalar('training/recall', tr_rec, epoch)
        log_writer.add_scalar('training/f1', 2*(tr_prec*tr_rec)/(tr_prec + tr_rec), epoch)

        log_writer.add_scalar('validation/loss', val_loss, epoch)
        log_writer.add_scalar('validation/precision', val_prec, epoch)
        log_writer.add_scalar('validation/recall', val_rec, epoch)
        log_writer.add_scalar('validation/f1', 2*(val_prec*val_rec)/(val_prec + val_rec), epoch)

        print("\n{}Train Loss: {}, Precision: {}, Recall: {}\033[0m".format(color, tr_loss, tr_prec, tr_rec))
        print("\n{}Val Loss: {}, Precision: {}, Recall: {}\033[0m".format(color, val_loss, val_prec, val_rec))

        if scheduler.num_of_reductions > train_parameters['max_reductions']:
            print('\033[93mTraining expired\033[0m')
            break

