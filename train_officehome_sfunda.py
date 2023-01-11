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
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix


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
    default="configs/officehome-train-config_OPDA.yaml",
    help="/path/to/config/file",
)
parser.add_argument("--dset", type=str, default="a2c")
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
parser.add_argument("--network", type=str, default="resnet50", help="network name")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=None, help="")
parser.add_argument("--no_adapt", default=False, action="store_true")
parser.add_argument("--save_model", default=False, action="store_true")
parser.add_argument("--lpa", default=False, action="store_true", help="use LPA")
parser.add_argument("--file", type=str, default="logfile")
parser.add_argument("--task", type=str, default="univ")
parser.add_argument("--model_name", type=str, default="slr_re")
parser.add_argument("--folder", type=str, default="weight")
parser.add_argument(
    "--save_path", type=str, default="record/ova_model", help="/path/to/save/model"
)
parser.add_argument(
    "--multi", type=float, default=0.1, help="weight factor for adaptation"
)

args = parser.parse_args()

ss = args.dset.split("2")[0]
tt = args.dset.split("2")[1]
if ss == "a":
    s = "Art"
elif ss == "c":
    s = "Clipart"
elif ss == "p":
    s = "Product"
elif ss == "r":
    s = "Real"

if tt == "a":
    t = "Art"
elif tt == "c":
    t = "Clipart"
elif tt == "p":
    t = "Product"
elif tt == "r":
    t = "Real"

args.source_data = "./data/source_{}_{}.txt".format(s, args.task)
args.target_data = "./data/target_{}_{}.txt".format(t, args.task)

current_folder = "./"
args.output_dir = osp.join(current_folder, args.folder, args.dset)
if not osp.exists(args.output_dir):
    os.system("mkdir -p " + args.output_dir)
if not osp.exists(args.output_dir):
    os.mkdir(args.output_dir)

args.output_file = osp.join(args.output_dir, "{}.txt".format(args.file))
with open(args.output_file, "w") as f:
    f.write(print_args(args) + "\n")
    f.flush()

if True:
    task = ["c", "a", "p", "r"]
task_s = args.dset.split("2")[0]
task.remove(task_s)
task_all = [task_s + "2" + i for i in task]
for task_sameS in task_all:
    path_task = os.getcwd() + "/" + args.folder + "/" + task_sameS
    if not osp.exists(path_task):
        os.mkdir(path_task)

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


print("num_class: " + str(num_class))


def get_model_mme(net, num_class=13, temp=0.05, top=False, norm=True):
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
    G, dim = get_model_mme(net, num_class=num_class)

    C = ResClassifier_MME(num_classes=num_class + 1, norm=False, input_size=dim)
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
    conf.train.min_step = 2000

    source_loader, target_loader, test_loader, target_folder = get_dataloaders(inputs)

    logname = log_set(inputs)

    G, C, opt_g, opt_c, param_lr_g, param_lr_c = get_models(inputs)
    ndata = target_folder.__len__()

    criterion = nn.CrossEntropyLoss().cuda()
    print("Task: %s train start!" % args.dset)
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)

    # source training
    # if True:
    if not osp.exists(
        osp.join(args.output_dir + "/source_F_single_{}.pt".format(args.model_name))
    ):
        # args.out_file = open(osp.join(args.output_dir, 'log_src_val.txt'), 'w')
        # args.out_file.write(print_args(args) + '\n')
        # args.out_file.flush()
        for step in range(conf.train.min_step):
            G.train()
            C.train()
            if step % len_train_source == 0:
                data_iter_s = iter(source_loader)
            data_s = next(data_iter_s)
            inv_lr_scheduler(
                param_lr_g,
                opt_g,
                step,
                init_lr=conf.train.lr,
                max_iter=conf.train.min_step,
            )
            inv_lr_scheduler(
                param_lr_c,
                opt_c,
                step,
                init_lr=conf.train.lr,
                max_iter=conf.train.min_step,
            )
            img_s = data_s[0]
            label_s = data_s[1]
            img_s, label_s = Variable(img_s.cuda()), Variable(label_s.cuda())
            opt_g.zero_grad()
            opt_c.zero_grad()
            C.module.weight_norm()

            # Source loss calculation
            feat = G(img_s)
            outs = C(feat)
            # source classification loss
            loss_s = nn.CrossEntropyLoss()(outs, label_s)

            p = [(i, j.item()) for i, j in enumerate(label_s)]
            outs_ = torch.cat(
                [torch.cat((outs[i][0:j], outs[i][j + 1 :])) for i, j in p]
            ).view(outs.shape[0], num_class)
            labels_unk = torch.LongTensor(
                [num_class - 1 for i in range(img_s.shape[0])]
            ).cuda()
            loss_unk = nn.CrossEntropyLoss()(outs_, labels_unk)

            all = loss_s + loss_unk

            all.backward()
            opt_g.step()
            opt_c.step()
            opt_g.zero_grad()
            opt_c.zero_grad()
            if (step > 0 and step % conf.test.test_interval == 0) or (
                step == conf.train.min_step - 1
            ):
                acc_o, known_acc, unknown, h_score = test_single(
                    step,
                    test_loader,
                    logname,
                    n_share,
                    G,
                    C,
                    args.output_file,
                    open=open,
                    num_class=num_class,
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
                    args.output_dir, "source_F_single_{}.pt".format(args.model_name)
                ),
            )
            torch.save(
                best_netC,
                osp.join(
                    args.output_dir, "source_C_single_{}.pt".format(args.model_name)
                ),
            )

        file_f = osp.join(
            args.output_dir + "/source_F_single_{}.pt".format(args.model_name)
        )
        file_c = osp.join(
            args.output_dir + "/source_C_single_{}.pt".format(args.model_name)
        )
        task.remove(args.dset.split("2")[1])
        task_remain = [task_s + "2" + i for i in task]
        for task_sameS in task_remain:
            path_task = os.getcwd() + "/" + args.folder + "/" + task_sameS
            pathF_copy = osp.join(
                path_task, "source_F_single_{}.pt".format(args.model_name)
            )
            pathC_copy = osp.join(
                path_task, "source_C_single_{}.pt".format(args.model_name)
            )
            shutil.copy(file_f, pathF_copy)
            shutil.copy(file_c, pathC_copy)

    else:
        print("Task already finished")
        G.load_state_dict(
            torch.load(
                osp.join(
                    args.output_dir, "source_F_single_{}.pt".format(args.model_name)
                )
            )
        )
        C.load_state_dict(
            torch.load(
                osp.join(
                    args.output_dir, "source_C_single_{}.pt".format(args.model_name)
                )
            )
        )

    acc_o, known_acc, unknown, h_score = test_single(
        s,
        test_loader,
        logname,
        n_share,
        G,
        C,
        output_file=args.output_file,
        open=open,
        num_class=num_class,
    )
    print(
        "Test source model task %s: known_acc: %s, unknown: %s, H value: %s "
        % (args.dset, known_acc, unknown, h_score)
    )

    # target adaptation

    # re-define param
    params = []
    for key, value in dict(G.named_parameters()).items():
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
    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])

    h_score_log = 0.0

    if args.lpa:
        loader = target_loader
        num_sample = len(loader.dataset)
        fea_bank = torch.randn(num_sample, 2048)
        score_bank = torch.randn(num_sample, num_class + 1).cuda()

        G.eval()
        C.eval()
        with torch.no_grad():
            iter_test = iter(loader)
            for i in range(len(loader)):
                data = iter_test.next()
                inputs_t = data[0]
                indx = data[-1]
                # labels = data[1]
                inputs_t = inputs_t.cuda()
                output = G(inputs_t)
                output_norm = F.normalize(output)
                outputs = C(output)
                outputs = nn.Softmax(-1)(outputs)

                fea_bank[indx] = output_norm.detach().clone().cpu()
                score_bank[indx] = outputs.detach().clone()  # .cpu()

    for step in range(conf.train.min_step):
        G.train()
        C.train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        data_t = next(data_iter_t)
        inv_lr_scheduler(
            param_lr_g, opt_g, step, init_lr=conf.train.lr, max_iter=conf.train.min_step
        )
        inv_lr_scheduler(
            param_lr_c, opt_c, step, init_lr=conf.train.lr, max_iter=conf.train.min_step
        )
        img_t = data_t[0]
        tar_idx = data_t[2]
        img_t = Variable(img_t.cuda())
        opt_g.zero_grad()
        opt_c.zero_grad()
        C.module.weight_norm()

        feat_t = G(img_t)
        out_open_t = C(feat_t)
        labels_pred = out_open_t.max(1)[1]
        out_softmax = nn.Softmax(dim=1)(out_open_t)

        nopred_unk = labels_pred != num_class
        labels_nounk = labels_pred[nopred_unk]
        out_open_t_nounk = out_open_t[nopred_unk]
        out_open_t_unk = out_open_t[labels_pred == num_class]

        all = 0

        prob_unk = out_softmax[:, -1]

        if args.lpa:
            with torch.no_grad():
                output_f_norm = F.normalize(feat_t)
                output_f_ = output_f_norm.cpu().detach().clone()

                pred_bs = out_softmax

                fea_bank[tar_idx] = output_f_.detach().clone().cpu()
                score_bank[tar_idx] = out_softmax.detach().clone()

                distance = output_f_ @ fea_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=3 + 1)
                idx_near = idx_near[:, 1:]  # batch x K
                score_near = score_bank[idx_near]  # batch x K x C

        if out_open_t_unk.shape[0] != 0:
            out_softmax_unk = out_softmax[labels_pred == num_class]
            ent_binary_unk = Entropy(out_open_t_unk)

            ent_binary_unk = torch.mean(ent_binary_unk)
            all += ent_binary_unk / out_open_t_unk.shape[0]

            if args.lpa:

                softmax_out_unk = out_softmax_unk.unsqueeze(1).expand(
                    -1, 3, -1
                )  # batch x K x C
                score_near_unk = score_near[labels_pred == num_class]
                loss_unk = torch.mean(
                    (
                        F.kl_div(softmax_out_unk, score_near_unk, reduction="none").sum(
                            -1
                        )
                    ).sum(1)
                )

                all += loss_unk / out_open_t_unk.shape[0]

        if labels_nounk.shape[0] != 0:
            out_softmax_nounk = out_softmax[nopred_unk]
            # ent_knw= torch.mean(Entropy(out_open_t_nounk))
            # all += ent_knw
            ent_binary_nounk = Entropy(out_open_t_nounk)

            ent_binary_nounk = torch.mean(ent_binary_nounk)
            all += ent_binary_nounk / labels_nounk.shape[0]

            if args.lpa:
                mask = torch.ones(
                    (out_softmax_nounk.shape[0], out_softmax_nounk.shape[0])
                )
                diag_num = torch.diag(mask)
                mask_diag = torch.diag_embed(diag_num)
                mask = mask - mask_diag
                copy = out_softmax_nounk[:, :-1].T  # .detach().clone()#
                dot_neg = out_softmax_nounk[:, :-1] @ copy  # batch x batch
                dot_neg = (dot_neg * mask.cuda()).sum(-1)
                neg_pred = torch.mean(dot_neg) / labels_nounk.shape[0]
                all += neg_pred  # * alpha

                softmax_out_un = out_softmax_nounk.unsqueeze(1).expand(
                    -1, 3, -1
                )  # batch x K x C
                score_near_nounk = score_near[nopred_unk]
                loss = torch.mean(
                    (
                        F.kl_div(
                            softmax_out_un, score_near_nounk, reduction="none"
                        ).sum(-1)
                    ).sum(1)
                )

                all += loss / labels_nounk.shape[0]

        all.backward()
        opt_g.step()
        # opt_c.step()
        opt_g.zero_grad()
        opt_c.zero_grad()
        if (step > 0 and step % conf.test.test_interval == 0) or (
            step == conf.train.min_step - 1
        ):
            acc_o, known_acc, unknown, h_score = test_single(
                step,
                test_loader,
                logname,
                n_share,
                G,
                C,
                output_file=args.output_file,
                open=open,
                num_class=num_class,
            )
            print(
                "Target adaptation task %s: known_acc: %s, unknown: %s, H value: %s "
                % (args.dset, known_acc, unknown, h_score)
            )

            G.train()
            C.train()
            if args.save_model:
                if h_score > h_score_log:
                    h_score_log = h_score
                    best_netF = G.state_dict()
                    best_netC = C.state_dict()
                    torch.save(
                        best_netF,
                        osp.join(
                            args.output_dir,
                            "target_F_single_{}.pt".format(args.model_name),
                        ),
                    )
                    torch.save(
                        best_netC,
                        osp.join(
                            args.output_dir,
                            "target_C_single_{}.pt".format(args.model_name),
                        ),
                    )


train_sfda()
