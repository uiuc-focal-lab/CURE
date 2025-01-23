import socket
import sys
import torch
import torch.nn as nn
import tqdm
from time import time
import torch.nn.functional as F

# hostname = socket.gethostname()
# if hostname == "dlsrlclarge.inf.ethz.ch" or hostname == "dlsrlplarge" or hostname == "dlsrlzlarge" or \
#         hostname == "dlsrltitan":
#     sys.path.append('/local/home/franziska/Git/unsound_provable_training')
# elif hostname == "dlsrlx":
#     sys.path.append('/home/franziska/Git/unsound_provable_training')
# elif hostname == "Zoo-MM":
#     sys.path.append('/home/mark/Projects/UPT/unsound_provable_training')

from src.AIDomains.wrappers import propagate_abs
from src.AIDomains.zonotope import HybridZonotope
from src.adv_attack import adv_whitebox, adv_whitebox_L2
from src.regularization import compute_bound_reg, compute_IBP_reg


def get_loss_FN(args):
    if args.loss_fn == "CE":
        loss_FN = nn.CrossEntropyLoss(reduction="none")
    elif args.loss_fn == "PT1":
        def loss_FN(pred, y):
            return F.cross_entropy(pred,y, reduction="none") + args.pt1_e * (1 - torch.gather(F.softmax(pred,1),1,y.unsqueeze(1))).squeeze(1)
    else:
        assert False, f"Loss function {args.loss_fn} is unknown."
    return loss_FN


def compute_regularization(args, net_abs, data, adex, eps, tau, max_tau, data_range):
    reg = torch.zeros(1, device=data.device)
    if args.cert_reg == "bound_reg" and tau < max_tau:
        if eps == 0.0:
            eps_reg = args.min_eps_reg
            data_abs = HybridZonotope.construct_from_noise(x=data, eps=eps_reg, domain="box", data_range=data_range)
            net_abs.reset_bounds()
            net_abs(data_abs)
        reg += compute_bound_reg(net_abs, eps, args.eps_end, reg_lambda=args.reg_lambda)
    elif args.cert_reg == "ibp_reg" and eps > 0.0:
        bs = data.shape[0]
        if args.box_attack == "concrete_attack":
            curr_eps = eps * 0.05  # TODO add tau for reg to args
            large_box = HybridZonotope.construct_from_noise(x=data, eps=eps, domain="box",
                                                            data_range=data_range)
            lb_large_box, ub_large_box = large_box.concretize()
            curr_midpoints = torch.clamp(adex, lb_large_box + curr_eps, ub_large_box - curr_eps)
            tiny_box = HybridZonotope.construct_from_noise(x=curr_midpoints, eps=curr_eps, domain="box",
                                                           data_range=data_range)
            net_abs(tiny_box)  # TODO not naive box but use prop?
        reg += compute_IBP_reg(net_abs, bs, args.reg_lambda)

    if args.l1 is not None:
        reg += args.l1 * sum([x.abs().sum() for x in net_abs.parameters()])

    return reg


def get_epsilon(args, eps_test, lambda_ratio, max_tau, lambda_scheduler, eps_scheduler, scheduler_index, train):
    if train:
        eps = eps_scheduler.getcurrent(scheduler_index)
    else:
        eps = eps_test

    if args.start_anneal_lambda is not None:
        lambda_ratio = lambda_scheduler.getcurrent(scheduler_index)
    else:
        lambda_ratio = lambda_ratio
    tau = lambda_ratio * eps

    if args.start_sound:  # while the full region is smaller than the final small region use the full region (during annealing)
        tau = min(max_tau, eps)

    return eps, tau


def get_propagation_region(args, net_abs, data, target, train, eps, tau, data_range, adv_step_size, adv_step_size_L2, adv_steps, dimwise_scaling):
    adex = None
    if train:
        if args.bn_mode_attack == "eval":
            net_abs.eval()  # use eval mode of BN for attack

        if args.box_attack == "pgd_concrete":
            net_abs.set_use_old_train_stats(True)
            # data_abs: the box with the center of adex
            # adex: adversarial example
            if args.L2_attack:
                adex, data_abs = adv_whitebox_L2(net_abs, data, target, tau, eps, n_steps=adv_steps, step_size=adv_step_size_L2,
                                            data_range=data_range, loss_function=args.box_attack_loss_fn, ODI_num_steps=0,
                                            restarts=1, train=True, dimwise_scaling=dimwise_scaling)
            else:
                adex, data_abs = adv_whitebox(net_abs, data, target, tau, eps, n_steps=adv_steps, step_size=adv_step_size,
                                            data_range=data_range, loss_function=args.box_attack_loss_fn, ODI_num_steps=0,
                                            restarts=1, train=True, dimwise_scaling=dimwise_scaling)

            net_abs.set_use_old_train_stats(False)
        elif args.box_attack == "centre": #  IBP training using smaller box 
            adex = data
            data_abs = HybridZonotope.construct_from_noise(x=data, eps=tau, domain="box", data_range=data_range,
                                                           dtype=data.dtype)
        else:
            assert False, f"box_attack: {args.box_attack} is unknown!"

        net_abs.train()

        if args.use_shrinking_box:
            shrinking_domain = args.shrinking_method + args.shrinking_relu_state
            data_abs.domain = shrinking_domain
            data_abs.c = args.shrinking_ratio
        if args.adv_bn:
            net_abs[0].set_track_running_stats(track_running_stats=False)
            midpoints = data_abs.get_head()
            net_abs(midpoints)
            net_abs[0].set_track_running_stats(track_running_stats=True)
    # training with standard IBP with larger box
    else:
        data_abs = HybridZonotope.construct_from_noise(x=data, eps=eps, domain="box", data_range=data_range)

    if args.bn and "concrete" in args.box_attack and train:
        net_abs[0].set_track_running_stats(track_running_stats=False)
        net_abs(data)
        net_abs[0].set_track_running_stats(track_running_stats=True)

    return data_abs, adex

# only call with joint training of Linf and L2 attacks using SABR
def get_propagation_region_joint(args, net_abs, data, target, train, eps, tau, eps_L2, tau_L2, data_range, adv_step_size, adv_step_size_L2, adv_steps, dimwise_scaling):
    # adex = None
    adex_l2, adex_linf = None, None
    if train:
        if args.bn_mode_attack == "eval":
            net_abs.eval()  # use eval mode of BN for attack

        if args.box_attack == "pgd_concrete":
            net_abs.set_use_old_train_stats(True)
            # data_abs: the box with the center of adex
            # adex: adversarial example

            adex_l2, data_abs_l2 = adv_whitebox_L2(net_abs, data, target, tau_L2, eps_L2, n_steps=adv_steps, step_size=adv_step_size_L2,
                                        data_range=data_range, loss_function=args.box_attack_loss_fn, ODI_num_steps=0,
                                        restarts=1, train=True, dimwise_scaling=dimwise_scaling)

            adex_linf, data_abs_linf = adv_whitebox(net_abs, data, target, tau, eps, n_steps=adv_steps, step_size=adv_step_size,
                                        data_range=data_range, loss_function=args.box_attack_loss_fn, ODI_num_steps=0,
                                        restarts=1, train=True, dimwise_scaling=dimwise_scaling)

            net_abs.set_use_old_train_stats(False)
        elif args.box_attack == "centre": #  IBP training using smaller box 
            adex_linf = data
            adex_l2 = data
            data_abs = HybridZonotope.construct_from_noise(x=data, eps=tau, domain="box", data_range=data_range,
                                                           dtype=data.dtype)
            data_abs_linf, data_abs_l2 = data_abs, data_abs
        else:
            assert False, f"box_attack: {args.box_attack} is unknown!"

        net_abs.train()

        if args.use_shrinking_box:
            shrinking_domain = args.shrinking_method + args.shrinking_relu_state
            data_abs_linf.domain = shrinking_domain
            data_abs_linf.c = args.shrinking_ratio
            # data_abs_l2.domain = shrinking_domain
            # data_abs_l2.c = args.shrinking_ratio
        if args.adv_bn:
            net_abs[0].set_track_running_stats(track_running_stats=False)
            # midpoints = data_abs.get_head()
            midpoints = (data_abs_linf.get_head() + data_abs_l2.get_head()) / 2.
            net_abs(midpoints)
            net_abs[0].set_track_running_stats(track_running_stats=True)
    # training with standard IBP with larger box
    else:
        data_abs = HybridZonotope.construct_from_noise(x=data, eps=eps, domain="box", data_range=data_range)
        data_abs_linf, data_abs_l2 = data_abs, data_abs

    if args.bn and "concrete" in args.box_attack and train:
        net_abs[0].set_track_running_stats(track_running_stats=False)
        net_abs(data)
        net_abs[0].set_track_running_stats(track_running_stats=True)

    return data_abs_linf, adex_linf, data_abs_l2, adex_l2

def train_net(net_abs, epoch, train, args, data_loader, input_dim, data_range, eps_test, use_cuda, adv_steps_scheduler, eps_scheduler, 
              eps_scheduler_L2 = None, clip_norm_scheduler=None, lambda_scheduler=None, kappa_scheduler=None, writer=None):

    # get epoch parameters from schedules
    if args.adv_end_steps is None:
        adv_steps = args.adv_start_steps
    else:
        adv_steps = int(args.adv_start_steps + (args.adv_end_steps - args.adv_start_steps) * adv_steps_scheduler.getcurrent(epoch))
    if args.adv_step_size_end is None:
        adv_step_size = args.adv_step_size
        adv_step_size_L2 = args.adv_step_size_L2
    else:
        adv_step_size = args.adv_step_size + (args.adv_step_size_end - args.adv_step_size) * adv_steps_scheduler.getcurrent(epoch)

    if args.end_clip_norm is not None:
        clip_norm = clip_norm_scheduler.getcurrent(epoch)
    else:
        clip_norm = args.clip_norm

    # lambda_ratio: tau / epsilon.
    max_tau = args.eps_end * max(args.lambda_ratio, args.end_lambda_ratio)

    if args.joint:
        max_tau_L2 = args.eps_end_L2 * max(args.lambda_ratio_L2, args.end_lambda_ratio)
    else:
        max_tau_L2 = None

    # Set up logging
    n_samples = 0
    nat_ok, abs_tau_ok, abs_eps_ok = 0, 0, 0
    abs_tau_ok_linf, abs_tau_ok_l2 = 0, 0
    loss_total, robust_tau_loss_total, robust_eps_loss_total, normal_loss_total, reg_loss_total = 0, 0, 0, 0, 0

    time_start = time()
    loss_FN = get_loss_FN(args)

    net_abs.eval()
    net_abs.set_dim(torch.rand((data_loader.batch_size, *input_dim), device="cuda" if use_cuda else "cpu"))
    if train:
        net_abs.train()
    else:
        net_abs.eval()

    pbar = tqdm.tqdm(data_loader)
    for batch_idx, (data, target, index) in enumerate(pbar):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # set the bounds to [-inf, inf], so we can recompute the bounds for the new data batch
        net_abs.reset_bounds()
        # Get batch parameters
        scheduler_index = epoch * len(data_loader) + batch_idx
        # eps, tau for the current batch
        eps, tau = get_epsilon(args, eps_test, args.lambda_ratio, max_tau, lambda_scheduler, eps_scheduler, scheduler_index, train)

        if args.joint:
            eps_L2, tau_L2 = get_epsilon(args, args.eps_test_L2, args.lambda_ratio_L2, max_tau_L2, lambda_scheduler, eps_scheduler_L2, scheduler_index, train)
        else:
            eps_L2, tau_L2 = None, None
        
        # kappa
        kappa = kappa_scheduler.getcurrent(scheduler_index)

        # net_abs.optimizer.zero_grad()

        out_normal = net_abs(data)
        # adversarial examples
        adex = None
        adex_linf, adex_l2 = None, None

        if not train or (kappa < 1.0 and tau > 0.0):
            # abstract propagation is needed for training or testing
            # the box region and adversarial example
            # jointly training two loss L1 + L2
            if args.joint:
                if args.random and train:
                    batch_size = len(target)
                    num_select = batch_size // 2

                    # Randomly select row indices
                    indices1 = torch.randperm(batch_size)[:num_select]
                    indices2 = torch.randperm(batch_size)[num_select:]

                    # Select rows
                    data1 = data[indices1]
                    data2 = data[indices2]

                    target1 = target[indices1]
                    target2 = target[indices2]

                    # get linf
                    data_abs_tau_linf, adex_linf = get_propagation_region(args, net_abs, data1, target1, train, eps, tau, data_range,
                                                            adv_step_size, adv_step_size_L2, adv_steps, args.dimwise_scaling)
                    net_abs.reset_bounds()
                    # only use box domain here, not sure about this step, how to get pseudo labels
                    out_abs_linf, pseudo_labels_linf = propagate_abs(net_abs, args.loss_domain, data_abs_tau_linf, target1)

                    # get l2
                    args.L2_attack = True
                    data_abs_tau_l2, adex_l2 = get_propagation_region(args, net_abs, data2, target2, train, eps_L2, tau_L2, data_range,
                                                            adv_step_size, adv_step_size_L2, adv_steps, args.dimwise_scaling)
                    net_abs.reset_bounds()
                    # only use box domain here, not sure about this step, how to get pseudo labels
                    out_abs_l2, pseudo_labels_l2 = propagate_abs(net_abs, args.loss_domain, data_abs_tau_l2, target2)

                    args.L2_attack = False

                    # Combine selected rows
                    adex_out_labels_random = torch.cat((out_abs_linf, out_abs_l2), dim=0)
                    adex_pseudo_labels_random = torch.cat((pseudo_labels_linf, pseudo_labels_l2), dim=0)

                    robust_loss = loss_FN(adex_out_labels_random, adex_pseudo_labels_random).mean()
                else:
                    data_abs_tau_linf, adex_linf, data_abs_tau_l2, adex_l2 = get_propagation_region_joint(args, net_abs, data, target, train, eps, tau, eps_L2, tau_L2, data_range,
                                                                adv_step_size, adv_step_size_L2, adv_steps, args.dimwise_scaling)
                    
                    net_abs.reset_bounds()
                    out_abs_linf, pseudo_labels_linf = propagate_abs(net_abs, args.loss_domain, data_abs_tau_linf, target)
                    out_abs_l2, pseudo_labels_l2 = propagate_abs(net_abs, args.loss_domain, data_abs_tau_l2, target)

                    # robust loss has two terms: avg - L1 + L2
                    # MAX
                    if args.max and train:
                        loss_linf = loss_FN(out_abs_linf, pseudo_labels_linf)
                        loss_l2 = loss_FN(out_abs_l2, pseudo_labels_l2)

                        tensor_list = [loss_linf, loss_l2]
                        out_labels_list = [out_abs_linf, out_abs_l2]
                        pseudo_labels_list = [pseudo_labels_linf, pseudo_labels_l2]
                        loss_arr = torch.stack(tuple(tensor_list))
                        out_labels_arr = torch.stack(tuple(out_labels_list))
                        pseudo_labels_arr = torch.stack(tuple(pseudo_labels_list))
                        max_loss = loss_arr.max(dim = 0)
                        pseudo_labels_best = pseudo_labels_arr[max_loss[1], torch.arange(len(target))]
                        out_labels_best = out_labels_arr[max_loss[1], torch.arange(len(target))]
                        robust_loss = loss_FN(out_labels_best, pseudo_labels_best).mean()
                        if args.reweight:
                            distance = adex_l2 - adex_linf
                            flattened_distance = distance.view(args.bs, -1)
                            distances = torch.norm(flattened_distance, dim=1, p=2) # shape [128]
                            robust_loss += distances.mean()
                    else: 
                        # default: AVG
                        robust_loss = (1 - args.alpha) * loss_FN(out_abs_linf, pseudo_labels_linf).mean() + args.alpha * loss_FN(out_abs_l2, pseudo_labels_l2).mean()

                # Logits pairing loss term over linf and l2
                if args.lp:
                    if args.all:
                        selected_kl = (out_abs_linf.argmax(1) == pseudo_labels_linf).detach()
                        out_linf_sel, out_l2_sel = out_abs_linf[selected_kl], out_abs_l2[selected_kl]

                        loss_kl_1 = 0

                        if len(out_linf_sel) > 0:
                            criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
                            loss_kl_1 = criterion_kl(F.log_softmax(out_l2_sel+1e-12, dim=1), F.softmax(out_linf_sel, dim=1)) / selected_kl.sum()

                        selected_kl = (out_abs_l2.argmax(1) == pseudo_labels_l2).detach()
                        out_linf_sel, out_l2_sel = out_abs_linf[selected_kl], out_abs_l2[selected_kl]

                        loss_kl_2 = 0

                        if len(out_linf_sel) > 0:
                            criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
                            loss_kl_2 = criterion_kl(F.log_softmax(out_linf_sel+1e-12, dim=1), F.softmax(out_l2_sel, dim=1)) / selected_kl.sum()
                        
                        robust_loss += (loss_kl_1 + loss_kl_2) * args.lbd
                        
                    elif args.reverse:
                        selected_kl = (out_abs_linf.argmax(1) == pseudo_labels_linf).detach()
                        out_linf_sel, out_l2_sel = out_abs_linf[selected_kl], out_abs_l2[selected_kl]

                        loss_kl = 0

                        if len(out_linf_sel) > 0:
                            criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
                            loss_kl = criterion_kl(F.log_softmax(out_l2_sel+1e-12, dim=1), F.softmax(out_linf_sel, dim=1)) / selected_kl.sum()
                        robust_loss += loss_kl * args.lbd
                    else:
                        selected_kl = (out_abs_l2.argmax(1) == pseudo_labels_l2).detach()
                        out_linf_sel, out_l2_sel = out_abs_linf[selected_kl], out_abs_l2[selected_kl]

                        loss_kl = 0

                        if len(out_linf_sel) > 0:
                            criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
                            loss_kl = criterion_kl(F.log_softmax(out_linf_sel+1e-12, dim=1), F.softmax(out_l2_sel, dim=1)) / selected_kl.sum()

                        robust_loss += loss_kl * args.lbd


                abs_tau_ok_linf += torch.eq(out_abs_linf.argmax(1), pseudo_labels_linf).sum()
                abs_tau_ok_l2 += torch.eq(out_abs_l2.argmax(1), pseudo_labels_l2).sum()
            else:
                data_abs_tau, adex = get_propagation_region(args, net_abs, data, target, train, eps, tau, data_range,
                                                            adv_step_size, adv_step_size_L2, adv_steps, args.dimwise_scaling)
                net_abs.reset_bounds()

                out_abs, pseudo_labels = propagate_abs(net_abs, args.loss_domain, data_abs_tau, target)

                robust_loss = loss_FN(out_abs, pseudo_labels).mean()
            

                abs_tau_ok += torch.eq(out_abs.argmax(1), pseudo_labels).sum()
        
        elif train and args.box_attack == "concrete_attack" and kappa < 1.0 and eps > 0.0:
            # adversarial loss for training - standard AT, tau = 0

            if args.bn_mode_attack == "eval":
                net_abs.eval()
            else:
                net_abs.set_use_old_train_stats(True)
            # perform normal pgd attack
            if args.L2_attack:
                adex, _ = adv_whitebox_L2(net_abs, data, target, 0.0, eps, n_steps=adv_steps, step_size=adv_step_size_L2,
                                   data_range=data_range, loss_function=args.box_attack_loss_fn, ODI_num_steps=0,
                                   restarts=1, train=True)
            else:
                adex, _ = adv_whitebox(net_abs, data, target, 0.0, eps, n_steps=adv_steps, step_size=adv_step_size,
                                    data_range=data_range, loss_function=args.box_attack_loss_fn, ODI_num_steps=0,
                                    restarts=1, train=True)

            if args.bn_mode_attack == "eval":
                # set status back to train
                net_abs.train()
                net_abs[0].set_track_running_stats(track_running_stats=False)
                out_adex = net_abs(adex)
                out_normal = net_abs(data)
                net_abs[0].set_track_running_stats(track_running_stats=True)
            else:
                out_adex = net_abs(adex)
                net_abs.set_use_old_train_stats(False)
            robust_loss = loss_FN(out_adex, target).mean()
            abs_tau_ok += torch.eq(out_adex.argmax(1), target).sum()
        else:
            robust_loss = torch.tensor(0.0)

        normal_loss = loss_FN(out_normal, target).mean()
        nat_ok += torch.eq(out_normal.argmax(1), target).sum()

        if train:
            net_abs.optimizer.zero_grad()
            if args.joint:
                reg = compute_regularization(args, net_abs, data, adex_linf, eps, tau, max_tau, data_range)
                # reg += compute_regularization(args, net_abs, data, adex_l2, eps_L2, tau_L2, max_tau_L2, data_range)
            else:
                if args.L2_attack:
                    reg = torch.tensor(0)
                else:
                    reg = compute_regularization(args, net_abs, data, adex, eps, tau, max_tau, data_range)

            robust_loss_scaled = (1 - kappa) * robust_loss
            normal_loss_scaled = kappa * normal_loss
            combined_loss = robust_loss_scaled + normal_loss_scaled + reg
            # combined_loss = robust_loss_scaled + normal_loss_scaled

            if args.clip_robust_gradient and robust_loss > 0.0:
                # clip only the robust loss
                robust_loss_scaled.backward()
                torch.nn.utils.clip_grad_norm_(net_abs.parameters(), clip_norm)
                (normal_loss_scaled + reg).backward()
            else:
                combined_loss.backward()
                if args.clip_combined_gradient is not None:
                    # clip both losses
                    torch.nn.utils.clip_grad_norm_(net_abs.parameters(), clip_norm)

            net_abs.optimizer.step()
            # torch.cuda.synchronize()
        else:
            combined_loss = (1 - kappa) * robust_loss + kappa * normal_loss
            reg = torch.tensor(0)

        time_epoch = time() - time_start

        reg_loss_total += reg.detach()
        robust_tau_loss_total += robust_loss.detach()
        normal_loss_total += normal_loss.detach()
        loss_total += combined_loss.detach()
        n_samples += target.size(0)

        if args.joint:
            description_str = f"[{epoch}:{batch_idx}:{'train' if train else 'test'}]: eps = [{tau:.6f}:{eps:.6f}], kappa={kappa:.3f}, loss nat: {normal_loss_total / (batch_idx + 1):.4f}, loss abs: {robust_tau_loss_total / (batch_idx + 1):.4f}, acc_nat={nat_ok / n_samples:.4f}, acc_abs_linf={abs_tau_ok_linf / n_samples:.4f}, acc_abs_l2={abs_tau_ok_l2 / n_samples:.4f}"
        else:
            description_str = f"[{epoch}:{batch_idx}:{'train' if train else 'test'}]: eps = [{tau:.6f}:{eps:.6f}], kappa={kappa:.3f}, loss nat: {normal_loss_total / (batch_idx + 1):.4f}, loss abs: {robust_tau_loss_total / (batch_idx + 1):.4f}, acc_nat={nat_ok / n_samples:.4f}, acc_abs={abs_tau_ok / n_samples:.4f}"
        pbar.set_description(description_str)
        pbar.refresh()

    ### Print such that logging picks it up
    print(description_str)

    # save metrics
    if args.save:
        if train:
            writer.add_scalar('kappa', kappa, epoch)
            writer.add_scalar('eps', eps, epoch)
            writer.add_scalar('tau', tau, epoch)
            writer.add_scalar('train_stand_acc', nat_ok / n_samples, epoch)
            if args.joint:
                writer.add_scalar('train_rob_acc_linf', abs_tau_ok_linf / n_samples, epoch)
                writer.add_scalar('train_rob_acc_l2', abs_tau_ok_l2 / n_samples, epoch)
            else:
                writer.add_scalar('train_rob_acc', abs_tau_ok / n_samples, epoch)
            writer.add_scalar('train_loss', loss_total / len(pbar), epoch)
            writer.add_scalar('train_normal_loss', normal_loss_total / len(pbar), epoch)
            writer.add_scalar('train_robust_loss', robust_tau_loss_total / len(pbar), epoch)
            writer.add_scalar('train_reg', reg / len(pbar), epoch)
            writer.add_scalar('train_time', time_epoch, epoch)
        else:
            writer.add_scalar('test_stand_acc', nat_ok / n_samples, epoch)
            if args.joint:
                writer.add_scalar('test_rob_acc_linf', abs_tau_ok_linf / n_samples, epoch)
                writer.add_scalar('test_rob_acc_l2', abs_tau_ok_l2 / n_samples, epoch)
            else:
                writer.add_scalar('test_rob_acc', abs_tau_ok / n_samples, epoch)
            writer.add_scalar('test_loss', loss_total / len(pbar), epoch)
            writer.add_scalar('test_normal_loss', normal_loss_total / len(pbar), epoch)
            writer.add_scalar('test_robust_loss', robust_tau_loss_total / len(pbar), epoch)
            writer.add_scalar('test_time', time_epoch, epoch)

def train_clean(net_abs, epoch, train, args, data_loader, input_dim, data_range, eps_test, use_cuda, adv_steps_scheduler, eps_scheduler, 
              eps_scheduler_L2 = None, clip_norm_scheduler=None, lambda_scheduler=None, kappa_scheduler=None, writer=None):

    # Set up logging
    n_samples = 0
    nat_ok, abs_tau_ok, abs_eps_ok = 0, 0, 0
    abs_tau_ok_linf, abs_tau_ok_l2 = 0, 0
    loss_total, robust_tau_loss_total, robust_eps_loss_total, normal_loss_total, reg_loss_total = 0, 0, 0, 0, 0

    time_start = time()
    loss_FN = get_loss_FN(args)

    net_abs.eval()
    net_abs.set_dim(torch.rand((data_loader.batch_size, *input_dim), device="cuda" if use_cuda else "cpu"))
    if train:
        net_abs.train()
    else:
        net_abs.eval()

    pbar = tqdm.tqdm(data_loader)
    for batch_idx, (data, target, index) in enumerate(pbar):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        out_normal = net_abs(data)

        normal_loss = loss_FN(out_normal, target).mean()
        nat_ok += torch.eq(out_normal.argmax(1), target).sum()

        combined_loss = normal_loss
        
        combined_loss.backward()
        net_abs.optimizer.step()

        time_epoch = time() - time_start
        normal_loss_total += normal_loss.detach()
        loss_total += combined_loss.detach()
        n_samples += target.size(0)

        description_str = f"[{epoch}:{batch_idx}:{'train' if train else 'test'}]: loss nat: {normal_loss_total / (batch_idx + 1):.4f}, acc_nat={nat_ok / n_samples:.4f}"
        pbar.set_description(description_str)
        pbar.refresh()

    ### Print such that logging picks it up
    print(description_str)

    return net_abs

