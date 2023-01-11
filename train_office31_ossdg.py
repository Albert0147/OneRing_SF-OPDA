from __future__ import print_function
import yaml
import easydict
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import log_set, save_model
from utils.lr_schedule import inv_lr_scheduler
from utils.defaults import get_dataloaders  # , get_models
from eval import *
import argparse
import torchvision.transforms as transforms
import torch.optim as optim
from data_loader.get_loader import get_loader, get_loader_label
import os.path as osp
import shutil
from models.basenet import *
import random


def Entropy(input_):
    input_ = nn.Softmax(dim=-1)(input_)
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


parser = argparse.ArgumentParser(
    description="Pytorch One Ring",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/office31_ossdg.yaml",
    help="/path/to/config/file",
)
parser.add_argument("--dset", type=str, default="a")
parser.add_argument(
    "--source_data",
    type=str,
    default="./utils/source_list.txt",
    help="path to source list",
)
parser.add_argument(
    "--target_data",
    type=str,
    default="./utils/target_list.txt",
    help="path to target list",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    help="how many batches before logging training status",
)
parser.add_argument(
    "--exp_name", type=str, default="office", help="/path/to/config/file"
)
parser.add_argument("--network", type=str, default="resnet18", help="network name")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=None, help="")
parser.add_argument("--no_adapt", default=False, action="store_true")
parser.add_argument("--save_model", default=False, action="store_true")
parser.add_argument("--file", type=str, default="logfile")
parser.add_argument("--model_name", type=str, default="ossdg31")
parser.add_argument(
    "--save_path", type=str, default="record/ova_model", help="/path/to/save/model"
)
parser.add_argument(
    "--multi", type=float, default=0.1, help="weight factor for adaptation"
)
parser.add_argument(
    "--decay", type=float, default=0.75, help="weight factor for adaptation"
)
parser.add_argument(
    "--alpha", type=float, default=0.5, help="weight factor for adaptation"
)
args = parser.parse_args()

ss = args.dset
if ss == "a":
    s = "amazon"
"""elif ss == 'c':
    s = 'Clipart'
elif ss == 'p':
    s = 'Product'
elif ss == 'r':
    s = 'Real_World'
    """


args.source_data = "./data/source_{}_obda.txt".format(s)
args.target_data = "./data/target_{}_ossdg.txt".format(s)

current_folder = "./"
args.output_dir = osp.join(current_folder, "weight_OSSDG31", args.dset)
if not osp.exists(args.output_dir):
    os.system("mkdir -p " + args.output_dir)
if not osp.exists(args.output_dir):
    os.mkdir(args.output_dir)

args.output_file = osp.join(args.output_dir, "{}.txt".format(args.file))
with open(args.output_file, "w") as f:
    f.write(print_args(args) + "\n")
    f.flush()


config_file = args.config
with open(config_file) as file_config:
    conf = yaml.load(file_config, Loader=yaml.FullLoader)
    save_config = yaml.load(file_config, Loader=yaml.FullLoader)
conf = easydict.EasyDict(conf)
gpu_devices = ",".join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
args.cuda = torch.cuda.is_available()

source_data = args.source_data
target_data = args.target_data
evaluation_data = args.target_data
network = args.network
use_gpu = torch.cuda.is_available()
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
n_total = conf.data.dataset.n_total
open = n_total - n_share - n_source_private > 0
num_class = n_share + n_source_private
script_name = os.path.basename(__file__)

inputs = vars(args)
inputs["evaluation_data"] = evaluation_data
inputs["conf"] = conf
inputs["script_name"] = script_name
inputs["num_class"] = num_class
inputs["config_file"] = config_file

SEED = 2022
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True


def get_model_mme(net, num_class=11, temp=0.05, top=False, norm=True):
    dim = 2048
    if "resnet" in net:
        model_g = ResBase(net, top=top)
        if "resnet18" in net:
            dim = 512
        if net == "resnet34":
            dim = 512
    elif "vgg" in net:
        model_g = VGGBase(option=net, pret=True, top=top)
        dim = 4096
    if top:
        dim = 1000
    print("selected network %s" % net)
    return model_g, dim


def get_models(kwargs):
    net = kwargs["network"]
    num_class = kwargs["num_class"]
    conf = kwargs["conf"]
    G, dim = get_model_mme(net, num_class=11)

    C = ResClassifier_MME(num_classes=11, norm=False, input_size=dim)
    device = torch.device("cuda")
    G.to(device)
    C.to(device)

    params = []
    if net == "vgg16":
        for key, value in dict(G.named_parameters()).items():
            if "classifier" in key:
                params += [
                    {
                        "params": [value],
                        "lr": conf.train.multi,
                        "weight_decay": conf.train.weight_decay,
                    }
                ]

    else:
        for key, value in dict(G.named_parameters()).items():

            if "bias" in key:
                params += [
                    {
                        "params": [value],
                        "lr": conf.train.multi,
                        "weight_decay": conf.train.weight_decay,
                    }
                ]
            else:
                params += [
                    {
                        "params": [value],
                        "lr": conf.train.multi,
                        "weight_decay": conf.train.weight_decay,
                    }
                ]
    opt_g = optim.SGD(
        params, momentum=conf.train.sgd_momentum, weight_decay=0.0005, nesterov=True
    )
    opt_c = optim.SGD(
        list(C.parameters()),
        lr=1.0,
        momentum=conf.train.sgd_momentum,
        weight_decay=0.0005,
        nesterov=True,
    )
    """[G, C1, C2], [opt_g, opt_c] = amp.initialize([G, C1, C2],
                                                  [opt_g, opt_c],
                                                  opt_level="O1")"""
    G = nn.DataParallel(G)
    C = nn.DataParallel(C)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])

    return G, C, opt_g, opt_c, param_lr_g, param_lr_c


def train_sfda():

    source_loader, target_loader, test_loader, target_folder = get_dataloaders(inputs)

    logname = log_set(inputs)

    G, C, opt_g, opt_c, param_lr_g, param_lr_c = get_models(inputs)
    ndata = target_folder.__len__()

    criterion = nn.CrossEntropyLoss().cuda()
    print("Source training: %s train start!" % args.dset)
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)

    # source training
    # if True:
    if not osp.exists(
        osp.join(args.output_dir + "/source_F_OSSDG_{}.pt".format(args.model_name))
    ):
        # args.out_file = open(osp.join(args.output_dir, 'log_src_val.txt'), 'w')
        # args.out_file.write(print_args(args) + '\n')
        # args.out_file.flush()
        for step in range(4000):
            G.train()
            C.train()
            if step % len_train_source == 0:
                data_iter_s = iter(source_loader)
            data_s = next(data_iter_s)
            inv_lr_scheduler(
                param_lr_g, opt_g, step, init_lr=conf.train.lr, max_iter=4000
            )
            inv_lr_scheduler(
                param_lr_c, opt_c, step, init_lr=conf.train.lr, max_iter=4000
            )
            img_s = data_s[0]
            label_s = data_s[1]
            img_s, label_s = Variable(img_s.cuda()), Variable(label_s.cuda())
            opt_g.zero_grad()
            opt_c.zero_grad()
            C.module.weight_norm()

            ## Source loss calculation
            feat = G(img_s)
            outs = C(feat)
            # out_open = C2(feat)
            ## source classification loss
            loss_s = nn.CrossEntropyLoss()(outs, label_s)

            p = [(i, j.item()) for i, j in enumerate(label_s)]
            outs_ = torch.cat(
                [torch.cat((outs[i][0:j], outs[i][j + 1 :])) for i, j in p]
            ).view(outs.shape[0], 10)
            labels_unk = torch.LongTensor([9 for i in range(img_s.shape[0])]).cuda()
            loss_unk = nn.CrossEntropyLoss()(outs_, labels_unk)

            all = loss_s + loss_unk  # * 0.5

            all.backward()
            opt_g.step()
            opt_c.step()
            opt_g.zero_grad()
            opt_c.zero_grad()
            if (step % conf.test.test_interval == 0) or (step == 3999):
                acc_o, known_acc, unknown, h_score = test_osdg31(
                    step,
                    target_loader,
                    logname,
                    n_share,
                    G,
                    C,
                    args.output_file,
                    open=open,
                )
                print(
                    "Source pretraining task %s: known_acc: %s, unknown: %s, H value: %s "
                    % (args.dset, known_acc, unknown, h_score)
                )
                G.train()
                C.train()
        if args.save_model:
            # save_path = "%s_%s_source.pth" % (args.save_path, step)
            # save_model(G, C1, C2, save_path)
            best_netF = G.state_dict()
            best_netC = C.state_dict()
            torch.save(
                best_netF,
                osp.join(
                    args.output_dir, "source_F_OSSDG_{}.pt".format(args.model_name)
                ),
            )
            torch.save(
                best_netC,
                osp.join(
                    args.output_dir, "source_C_OSSDG_{}.pt".format(args.model_name)
                ),
            )

    else:
        print("Task already finished")
        G.load_state_dict(
            torch.load(
                osp.join(
                    args.output_dir, "source_F_OSSDG_{}.pt".format(args.model_name)
                )
            )
        )
        C.load_state_dict(
            torch.load(
                osp.join(
                    args.output_dir, "source_C_OSSDG_{}.pt".format(args.model_name)
                )
            )
        )

    acc_o, known_acc, unknown, h_score = test_osdg31(
        s, test_loader, logname, n_share, G, C, output_file=args.output_file, open=open
    )
    print(
        "Test source model task %s: known_acc: %s, unknown: %s, H value: %s "
        % (args.dset, known_acc, unknown, h_score)
    )


train_sfda()
