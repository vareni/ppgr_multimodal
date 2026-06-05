import torch
import torch.nn as nn
from .myresnet import resnet101, Resnet101_concat
from collections import OrderedDict

from .CGM_models import CGMHead, CGMHeadAttention, CGMHeadExpanded, CGMHeadAttentionMicroFiLM


class MacroPlusCGM(nn.Module):
    def __init__(self, macro_net: list[nn.Module], cgm_head: nn.Module, detach_macros: bool = False,
                 train_macro: bool = False, n_macros: int = 4, latent_macro_dim: int = 32):
        super().__init__()
        self.net_rgb = macro_net[0]
        self.net_depth = macro_net[1]
        self.net_cat = macro_net[2]
        self.cgm_head = cgm_head
        self.detach_macros = detach_macros
        self.train_macro = train_macro

        self.macro_norm = nn.BatchNorm1d(n_macros)
        self.macro_latent_norm = nn.BatchNorm1d(latent_macro_dim)

        self.macro_decode_heads = nn.ModuleList([
            nn.Linear(latent_macro_dim, 1) for _ in range(n_macros)
        ])

        # assert not (train_macro and detach_macros), "Train macro and detach macros can't go together!!!"

        if not train_macro:
            for p in self.net_rgb.parameters():
                p.requires_grad = False
            for p in self.net_depth.parameters():
                p.requires_grad = False
            for p in self.net_cat.parameters():
                p.requires_grad = False
            # for name, p in self.net_cat.named_parameters():
            #     if "latent_proj" in name:
            #         p.requires_grad = True
            #         print("Training latent_proj layer in net_cat")
            #     else:
            #         p.requires_grad = False

    def forward(self, tab_feats, img=None, img_depth=None, true_macros=None):
        # TODO debug
        if true_macros is not None:
            # macros_normed = self.macro_norm(true_macros.float()) if true_macros.size(0) > 1 else true_macros
            x = torch.cat([true_macros, tab_feats], dim=-1)
            logits = self.cgm_head(x)
            return logits  # outputs, macros

        # your macro net returns a list/tuple of outputs; in direct_prediction you used outputs[0..4]
        p2, p3, p4, p5 = self.net_rgb(img)
        outputs_rgbd = self.net_depth(img_depth)
        d2, d3, d4, d5 = outputs_rgbd
        # outputs = self.net_cat([p2, p3, p4, p5], [d2, d3, d4, d5])
        #
        # macros_raw = torch.stack([outputs[0], outputs[2], outputs[3], outputs[4]], dim=-1)  #outputs[1] - mass
        # macros_cgm = self.macro_norm(macros_raw) if macros_raw.size(0) > 1 else macros_raw
        #
        # if self.detach_macros:
        #     macros_cgm = macros_cgm.detach()
        #
        # x = torch.cat([macros_cgm, tab_feats], dim=-1)

        img_latent = self.net_cat([p2, p3, p4, p5], [d2, d3, d4, d5])  # (B, 32)

        pred_macros = torch.cat(
            [head(img_latent) for head in self.macro_decode_heads], dim=-1
        )  # (B, n_macros)

        img_latent = self.macro_latent_norm(img_latent) if img_latent.size(0) > 1 else img_latent

        if self.detach_macros:
            img_latent = img_latent.detach()

        x = torch.cat([img_latent, tab_feats], dim=-1)

        logits = self.cgm_head(x)
        return logits, pred_macros #macros_raw  #outputs, macros

    def train(self, mode=True):
        super().train(mode)
        # self.cgm_head.train(mode)

        if not self.train_macro:
            self.net_rgb.eval()
            self.net_depth.eval()
            self.net_cat.eval()
        return self

def load_macro_models(args, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = resnet101(rgbd=args.rgbd)
    net2 = resnet101(rgbd=args.rgbd)
    net_cat = Resnet101_concat()

    if args.resume:
        print('==> Resuming from checkpoint..')
        models_state_dict = torch.load(args.resume)

        net.load_state_dict(models_state_dict["net"])

        new_state_dict_depth = OrderedDict()
        for k, v in models_state_dict['net_d'].items():
            name = k[7:] if k.startswith('module') else k
            new_state_dict_depth[name] = v
        missing_keys, _ = net2.load_state_dict(new_state_dict_depth)

        new_state_dict_cat = OrderedDict()
        for k, v in models_state_dict['net_cat'].items():
            name = k[7:] if k.startswith('module') else k
            new_state_dict_cat[name] = v
        net_cat.load_state_dict(new_state_dict_cat)

        # optimizer.load_state_dict(models_state_dict['optimizer'])
        # start_epoch = models_state_dict['epoch']
        return net, net2, net_cat

    print('==> Load checkpoint..')
    resnet101_food2k = torch.load("./CHECKPOINTS/food2k_resnet101_0.0001.pth")
    pretrained_dict = resnet101_food2k

    model_dict = net.state_dict()
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if k in model_dict: #update the same part
            name = k[7:] if k.startswith('module') else k
            new_state_dict[name] = v
    model_dict.update(new_state_dict)
    net.load_state_dict(model_dict)
    if args.rgbd:
        net2.load_state_dict(model_dict)

    net = net.to(device)
    if args.rgbd:
        net2 = net2.to(device)
        net_cat = net_cat.to(device)

    return net, net2, net_cat


def load_partial_pretrained_macro_weights(args, device):
    """
    Load net_rgb, net_depth fully, and net_cat partially
    (shared conv/attention weights only, skip old macro regression heads).
    Freezes everything except net_cat.latent_proj and cgm_head.
    """
    net = resnet101(rgbd=args.rgbd)
    net2 = resnet101(rgbd=args.rgbd)
    net_cat = Resnet101_concat(latent_dim=args.latent_macro_dim)

    checkpoint_path = args.resume
    if not checkpoint_path:
        print(f"==> Loading macro weights from food2k")
        pretrained_dict = torch.load("./CHECKPOINTS/food2k_resnet101_0.0001.pth")
        model_dict = net.state_dict()
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if k in model_dict:  # update the same part
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
        model_dict.update(new_state_dict)
        net.load_state_dict(model_dict)
        net2.load_state_dict(model_dict)

        return net, net2, net_cat

    print(f"==> Loading pretrained macro weights from {checkpoint_path}")
    models_state_dict = torch.load(checkpoint_path, map_location=device)

    net.load_state_dict(models_state_dict["net"])

    new_state_dict_depth = OrderedDict()
    for k, v in models_state_dict['net_d'].items():
        name = k[7:] if k.startswith('module') else k
        new_state_dict_depth[name] = v
    missing_keys, _ = net2.load_state_dict(new_state_dict_depth)

    new_state_dict_cat = OrderedDict()
    for k, v in models_state_dict['net_cat'].items():
        name = k[7:] if k.startswith('module') else k
        new_state_dict_cat[name] = v

    current_net_cat_state = net_cat.state_dict()

    # Only load keys that exist in both and have matching shapes
    matched = {
        k: v for k, v in new_state_dict_cat.items()
        if k in current_net_cat_state and current_net_cat_state[k].shape == v.shape
    }
    skipped = [k for k in new_state_dict_cat if k not in matched]
    new_keys = [k for k in current_net_cat_state if k not in new_state_dict_cat]

    current_net_cat_state.update(matched)
    net_cat.load_state_dict(current_net_cat_state)
    print(f"net_cat: loaded {len(matched)} tensors, "
          f"skipped old keys {skipped}, "
          f"new random-init keys {new_keys}")

    return net, net2, net_cat


def choose_CGM_model(args, in_dim, out_dim, device):
    n_micro = 0
    if args.microbiome:
        n_micro = 5

    if args.cgm_model == 'CGMHead':
        return CGMHead(in_dim=in_dim, out_dim=out_dim).to(device)  # iAUC - out=1
    elif args.cgm_model == 'CGMHeadAttention':
        return CGMHeadAttention(in_dim=in_dim, out_dim=out_dim).to(device)
    elif args.cgm_model == 'CGMHeadExpanded':
        return CGMHeadExpanded(in_dim=in_dim, out_dim=out_dim).to(device)
    elif args.cgm_model == 'CGMHeadAttentionMicroFiLM':
        return CGMHeadAttentionMicroFiLM(in_dim=in_dim, out_dim=out_dim, n_micro=n_micro,
                                         n_macros=args.latent_macro_dim).to(device)
    raise ValueError('Unrecognized CGM model')


def assemble_joint_model(args, device=None, out_dim=2):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.microbiome:
        # in_dim = 4 + 15 + 5 + 1  # macros + glucose + microbiome + fiber
        in_dim = args.latent_macro_dim + 15 + 5  # macros + glucose + microbiome
    else:
        # in_dim = 4 + 15 + 1  # macros + glucose + fiber
        in_dim = args.latent_macro_dim + 15  # macros + glucose

    # cgm_head = CGMHead(in_dim=in_dim, hidden=64, out_dim=out_dim).to(device)  # iAUC - out=1

    cgm_head = choose_CGM_model(args, in_dim, out_dim, device)

    # net, net2, net_cat = load_macro_models(args, device)
    net, net2, net_cat = load_partial_pretrained_macro_weights(args, device)
    macro_net = [net, net2, net_cat]

    joint_model = MacroPlusCGM(macro_net, cgm_head, detach_macros=False, train_macro=args.train_macro,
                               latent_macro_dim=args.latent_macro_dim).to(device)

    if device == 'cuda':
        joint_model = torch.nn.DataParallel(joint_model)

    return joint_model

def load_joint_model(checkpoint_path, device=None, microbiome=False, out_dim=2, cgm_model='CGMHead'):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net_rgb = resnet101(rgbd=True)
    net_depth = resnet101(rgbd=True)
    net_cat = Resnet101_concat()

    if microbiome:
        in_dim = 4 + 15 + 5 + 1  # macros + glucose + microbiome + fiber
    else:
        in_dim = 4 + 15 + 1  # macros + glucose + fiber

    # cgm_head = CGMHead(in_dim=in_dim, hidden=64, out_dim=out_dim).to(device)
    n_micro = 0
    if microbiome:
        n_micro = 5
    cgm_head = choose_CGM_model(cgm_model, in_dim, out_dim, device, n_micro)

    joint_model = MacroPlusCGM([net_rgb, net_depth, net_cat], cgm_head, detach_macros=True).to(device)

    models_state_dict = torch.load(checkpoint_path)

    joint_model.net_rgb.load_state_dict(models_state_dict["net"])
    joint_model.net_depth.load_state_dict(models_state_dict["net_d"])
    joint_model.net_cat.load_state_dict(models_state_dict["net_cat"])
    joint_model.cgm_head.load_state_dict(models_state_dict["cgm_head"])

    joint_model.macro_norm.load_state_dict(models_state_dict["macro_norm"])

    joint_model.to(device)
    joint_model.eval()

    print(f'Loaded best model from epoch: {models_state_dict["epoch"]}')

    return joint_model
