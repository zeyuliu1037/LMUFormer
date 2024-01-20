import argparse
import os
import ast
import sys
import time
import yaml
import torch
from datetime import datetime
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from timm.models import create_model
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import numpy as np
import lmu_rnn
from traintest import train, validate
from utilities import *

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='gsc.yml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument("--exp-dir", type=str, default="./logs", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

parser.add_argument("--dataset_mean", type=float, default=-4.2677393, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, help="the dataset spectrogram std")
parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')

parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
parser.add_argument("--lrscheduler_start", type=int, default=2, help="which epoch to start reducing the learning rate")
parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")

parser.add_argument('--wa', help='if weight averaging', type=ast.literal_eval, default='False')
parser.add_argument('--wa_start', type=int, default=1, help="which epoch to start weight averaging the checkpoint model")
parser.add_argument('--wa_end', type=int, default=5, help="which epoch to end weight averaging the checkpoint model")

parser.add_argument('-T', '--time-step', type=int, default=4, metavar='time',
                    help='simulation time step of spiking neuron (default: 4)')
parser.add_argument('-L', '--layer', type=int, default=4, metavar='layer',
                    help='model layer (default: 4)')
parser.add_argument('--dim', type=int, default=None, metavar='N',
                    help='embedding dimsension of feature')
parser.add_argument('--num_heads', type=int, default=None, metavar='N',
                    help='attention head number')
parser.add_argument('--patch-size', type=int, default=None, metavar='N',
                    help='Image patch size')
parser.add_argument('--mlp-ratio', type=int, default=None, metavar='N',
                    help='expand ration of embedding dimension in MLP block')


parser.add_argument('--gpu_id', type=str, default='4',
                    help='the id of the gpu(for one gpu)')
parser.add_argument('--annotaiton', type=str, default='',
                    help='annotaiton for exps')
parser.add_argument('--test-only', action='store_true', default=False,
                    help='if only test on the validation and test dataset')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--test_mode', type=str, default='normal',
                    help='if use recurrent forward in final validation and test, normal or recurrent')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

if __name__ == '__main__':
    args, args_text = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Number of GPUs available:", torch.cuda.device_count(), ', device: ', device)

    audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
                'noise':args.noise}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':False}

    if args.bal == 'bal':
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # transformer based model    
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.n_class, img_size_h=128, img_size_w=args.audio_length,
        patch_size=args.patch_size, embed_dims=args.dim, num_heads=args.num_heads, mlp_ratios=args.mlp_ratio,
        in_channels=args.in_channels, qkv_bias=False,
        depths=args.layer, sr_ratios=1,
        T=args.time_step,
        checkpoint_path=args.initial_checkpoint,
        test_mode=args.test_mode
        )    

    if args.loss == 'BCE':
        args.loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        args.loss_fn = nn.CrossEntropyLoss()
        
    if not args.test_only:
        args.exp_dir = args.exp_dir + '/' + args.model + '_gc_' + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(args.exp_dir, exist_ok=True)
        os.makedirs(args.exp_dir + '/models', exist_ok=True)
        print("\nCreating experiment directory: %s" % args.exp_dir)
        with open(os.path.join(args.exp_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
            f.write('\n{}'.format(model))
        f = open(os.path.join(args.exp_dir, 'log.txt'), 'w')
        f.write('\nNow starting training for {:d} epochs'.format(args.n_epochs))

        train(model, train_loader, val_loader, args, f, device)
    else:
        directory = os.path.dirname(args.initial_checkpoint)
        args.exp_dir = os.path.dirname(directory)
        f = open(os.path.join(args.exp_dir, 'log.txt'), 'a')

    # for speechcommands dataset, evaluate the best model on validation set on the test set
    if args.dataset == 'speechcommands':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(args.exp_dir + '/models/best_model.pth', map_location=device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(sd)

        # best model on the validation set
        stats, _, check_test_acc = validate(model, val_loader, args, 'valid_set', device, args.test_mode)

        # note it is NOT mean of class-wise accuracy
        val_acc = stats[0]['acc']
        val_mAUC = np.mean([stat['auc'] for stat in stats])
        f.write('\n\n---------------evaluate on the validation set---------------')
        f.write("\nAccuracy: {:.6f}".format(val_acc))
        f.write('\nchech_val_acc: {:.4f}'.format(check_test_acc))
        f.write("\nAUC: {:.6f}".format(val_mAUC))

        # test the model on the evaluation set
        eval_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        stats, _, check_test_acc = validate(model, eval_loader, args, 'eval_set', device, args.test_mode)

        eval_acc = stats[0]['acc']
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        f.write('\n---------------evaluate on the test set---------------')
        f.write("\nAccuracy: {:.6f}".format(eval_acc))
        f.write('\nchech_test_acc: {:.4f}'.format(check_test_acc))
        f.write("\nAUC: {:.6f}".format(eval_mAUC))
        np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])

