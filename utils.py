import os
import torch
import yaml
import torch.nn as nn
import parser
from model import *


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s' % dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


######################################################################
# Save model
# ---------------------------
def save_network(network, dirname, epoch_label):
    if not os.path.isdir('./model/' + dirname):
        os.mkdir('./model/' + dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model',dirname,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
#  Load model for resume
# ---------------------------
def load_network(name, opt):
    # Load config
    dirname = os.path.join('./model',name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    print(last_model_name,"last_model_name")  # net_119.pth last_model_name

    if opt.epoch is not None:
        epoch = opt.epoch
        print(epoch,"use epoch")
    else:
        epoch = last_model_name.split('_')[1]
        epoch = epoch.split('.')[0]
    if not epoch == 'last':
        epoch = int(epoch)
    config_path = os.path.join(dirname, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    opt.infonce = config['infonce']
    opt.name = config['name']
    opt.data_dir = config['data_dir']
    opt.train_all = config['train_all']
    opt.droprate = config['droprate']
    opt.color_jitter = config['color_jitter']
    opt.batchsize = config['batchsize']
    opt.h = config['h']
    opt.w = config['w']
    opt.share = config['share']
    opt.stride = config['stride']
    if 'LPN' in config:
        opt.LPN = config['LPN']
    else:
        opt.LPN = False
    if 'pool' in config:
        opt.pool = config['pool']
    if 'h' in config:
        opt.h = config['h']
        opt.w = config['w']
    if 'gpu_ids' in config:
        opt.gpu_ids = config['gpu_ids']
    opt.erasing_p = config['erasing_p']
    opt.lr = config['lr']
    opt.nclasses = config['nclasses']
    opt.erasing_p = config['erasing_p']
    opt.use_dense = config['use_dense']
    opt.fp16 = config['fp16']
    opt.views = config['views']
    if 'block' in config:
        opt.block = config['block']
    else:
        opt.block = 1
    if 'swin' in config:
        opt.swin = config['swin']
    else:
        opt.swin = False

    if opt.use_dense:
        model = ft_net_dense(opt.nclasses, opt.droprate, opt.stride, None, opt.pool)
    # if opt.LPN:
    #     model = LPN(opt.nclasses)
    if opt.views == 2:
        opt.use_vgg16 = config['use_vgg16']
        model = two_view_net(opt.nclasses, droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                             share_weight=opt.share,
                             VGG16=opt.use_vgg16, LPN=opt.LPN, block=opt.block, swin=opt.swin)
    else:
        if opt.LPN:
            model = CSWinTransv2_threeIn(opt.nclasses, droprate=opt.droprate, decouple=False, infonce=opt.infonce)

        elif opt.swin:
            model = CSWinTransv2_threeIn(opt.nclasses, droprate=opt.droprate)
        else:
            if opt.modelNum == 1:
                # 需要实时的进行修改
                model = CSWinTransv2_threeIn(opt.nclasses, droprate=opt.droprate, decouple=False)
            else:
                model = CSWinTransv2_threeIn(opt.nclasses, droprate=opt.droprate,decouple = opt.decouple)
            # model = CSWinTransv2_threeIn(opt.nclasses, droprate=opt.droprate, stride=opt.stride, pool=opt.pool)
    # print(model)
    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth' % epoch
    else:
        save_filename = 'net_%s.pth' % epoch
    # save_filename = 'net_099.pth'
    save_path = os.path.join('./model',name,save_filename)
    print('Load the model from %s' % save_path)
    network = model
    # network.load_state_dict(torch.load(save_path))
    try:
        network.load_state_dict(torch.load(save_path))
    except:
        network = torch.nn.DataParallel(network)
        network.load_state_dict(torch.load(save_path))
        network = network.module

    return network, opt, epoch


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    toogle_grad(model_src, True)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # print(self.state[p]["old_p"],"1233333333333")
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels_column = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            #mask = torch.eq(labels, labels.T).float().to(device)
            mask = (labels_column == labels).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # print(contrast_count,"contrast_count")
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(contrast_feature.shape,"contrast_feature")

        # print(f"contrast_feature {contrast_feature.size()}")
        # print(f"mask {mask.size()}")
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print(f"mask repeat {mask.size()}")
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # print(f"logits {logits.size()}")
        # assert(0)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss