import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
from torch.nn.utils import clip_grad_norm_
from spikingjelly.clock_driven import functional

def train(model, train_loader, test_loader, args, f, device):
    f.write('\nrunning on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    train_acc = AverageMeter()
    progress = []
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    model = model.to(device)
    # Set up the optimizer
    trainables = [p for p in model.parameters() if p.requires_grad]
    f.write('\nTotal parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    f.write('\nTotal trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=0, betas=(0.95, 0.999))

    # dataset specific settings
    main_metrics = args.metrics
    warmup = args.warmup
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    loss_fn = args.loss_fn
    f.write('\nnow training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    f.write('\nThe learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epochs'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    epoch += 1
    # for amp
    scaler = GradScaler()

    f.write("\ncurrent #steps=%s, #epochs=%s" % (global_step, epoch))
    f.write("\nstart training...")
    result = np.zeros([args.n_epochs, 10])
    model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        f.write('\n---------------\n')
        f.write(''.format(datetime.datetime.now()))
        f.write("\ncurrent #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, labels) in enumerate(train_loader):

            B = audio_input.size(0)

            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time, B)
            per_sample_data_time.update((time.time() - end_time) / B, B)
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                f.write('\nwarm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            with autocast():
                model.module.act_loss = 0
                audio_output = model(audio_input)
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    ce_loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
                else:
                    ce_loss = loss_fn(audio_output, labels)
                # print('shape:', labels.shape) # [B, 35]
            acc = (torch.argmax(audio_output, axis=1) == torch.argmax(labels.long(), axis=1)).sum() / B

            # optimization if amp is not used
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # record loss
            loss_meter.update(ce_loss.item(), B)
            
            # optimiztion if amp is used
            loss = ce_loss
            # print('loss: ', loss.item(), 'act_loss: ', model.module.act_loss.item()*l1_reg_coef)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # for name, param in model.named_parameters():
            #     if ('A' in name or 'B' in name)and param.grad is not None:
            #         # print(name, param.grad.norm())
            #         clip_grad_norm_(param, 1)
            # exit()
            scaler.step(optimizer)
            scaler.update()
            functional.reset_net(model.module)

            train_acc.update(acc.item(), B)
            batch_time.update(time.time() - end_time, B)
            per_sample_time.update((time.time() - end_time)/B, B)
            per_sample_dnn_time.update((time.time() - dnn_start_time)/B, B)

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                f.write('\nEpoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'
                  'Train Acc {train_acc:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, 
                      loss_meter=loss_meter, 
                      train_acc=acc.item()))
                if np.isnan(loss_meter.avg):
                    f.write("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        train_time=time.time()-begin_time
        f.write(f'\nAvg training acc: {train_acc.avg:.4f}, training time: {train_time:.2f}'.format(train_acc=train_acc, train_time=train_time))

        f.write('\nstart validation\n')
        val_start_time = time.time()
        stats, valid_loss, check_test_acc = validate(model, test_loader, args, epoch, device)
        f.write('\nvalidation time: {:.3f}'.format(time.time()-val_start_time))
        # ensemble results
        cum_stats = validate_ensemble(args, epoch)
        cum_mAP = np.mean([stat['AP'] for stat in cum_stats])
        cum_mAUC = np.mean([stat['auc'] for stat in cum_stats])
        cum_acc = cum_stats[0]['acc']

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        if main_metrics == 'mAP':
            f.write("mAP: {:.6f} ".format(mAP))
        else:
            f.write("acc: {:.6f} ".format(acc))
        f.write("AUC: {:.6f} ".format(mAUC))
        f.write("Avg Precision: {:.6f} ".format(average_precision))
        f.write("Avg Recall: {:.6f} ".format(average_recall))
        f.write("d_prime: {:.6f} ".format(d_prime(mAUC)))
        f.write("train_loss: {:.6f} ".format(loss_meter.avg))
        f.write("valid_loss: {:.6f} ".format(valid_loss))
        f.write("check test acc: {:.6f} ".format(check_test_acc))

        if main_metrics == 'mAP':
            result[epoch-1, :] = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, cum_mAP, cum_mAUC, optimizer.param_groups[0]['lr']]
        else:
            result[epoch-1, :] = [acc, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, cum_acc, cum_mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        f.write('\nvalidation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if cum_mAP > best_cum_mAP:
            best_cum_epoch = epoch
            best_cum_mAP = cum_mAP

        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

            with open(exp_dir + '/stats_best' + '.pickle', 'wb') as handle:
                pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

        scheduler.step()

        f.write('\nEpoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        
        _save_progress()

        finish_time = time.time()
        f.write('\nepoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

    # if args.dataset == 'audioset':
    #     if len(train_loader.dataset) > 2e5:
    #         stats=validate_wa(model, test_loader, args, 1, 5)
    #     else:
    #         stats=validate_wa(model, test_loader, args, 6, 25)
    if args.wa == True:
        stats = validate_wa(model, test_loader, args, args.wa_start, args.wa_end, device)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)
        wa_result = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC)]
        f.write('\n---------------Training Finished---------------')
        f.write('\nweighted averaged model results')
        f.write("\nmAP: {:.6f}".format(mAP))
        f.write("\nAUC: {:.6f}".format(mAUC))
        f.write("\nAvg Precision: {:.6f}".format(average_precision))
        f.write("\nAvg Recall: {:.6f}".format(average_recall))
        f.write("\nd_prime: {:.6f}".format(d_prime(mAUC)))
        f.write("\ntrain_loss: {:.6f}".format(loss_meter.avg))
        f.write("\nvalid_loss: {:.6f}".format(valid_loss))
        np.savetxt(exp_dir + '/wa_result.csv', wa_result)

def validate(model, val_loader, args, epoch, device, mode='normal'):
    batch_time = AverageMeter()
    test_acc = AverageMeter()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    # switch to evaluate mode
    model.eval()

    end = time.time()
    A_predictions = [[] for _ in range(128)] if mode == 'all_seq' else []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            B = labels.shape[0]
            audio_input = audio_input.to(device)

            # compute output
            # print('val: ', audio_input.shape)
            if mode == 'all_seq':
                audio_output = model(audio_input)
                N,_,_ = audio_output.shape
                all_audio_output = audio_output
                for n in range(N):
                    all_audio_output[n,:,:] = torch.sigmoid(audio_output[n,:,:])
                audio_output = all_audio_output[-1,:,:]
                predictions = all_audio_output.to('cpu').detach()
                for n in range(N):
                    A_predictions[n].append(predictions[n,:,:])
            else:
                audio_output = model(audio_input)
            
                audio_output = torch.sigmoid(audio_output)
                predictions = audio_output.to('cpu').detach()
                A_predictions.append(predictions)

            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end, B)
            end = time.time()
            functional.reset_net(model)

            acc = (torch.argmax(audio_output, axis=1) == torch.argmax(labels.long(), axis=1)).sum().item() / B
            test_acc.update(acc, B)

        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        if mode == 'all_seq':
            acc_list = []
            for n in range(128):
                audio_output_n = torch.cat(A_predictions[n])
                stats = calculate_stats(audio_output_n, target)
                acc_list.append(stats[0]['acc'])
            with open('acc_list.pkl', 'wb') as f:
                print('saving acc_list.pkl')
                pickle.dump(acc_list, f)
                print('acc_list.pkl saved')
            audio_output = audio_output_n
        else:
            audio_output = torch.cat(A_predictions)
            target = torch.cat(A_targets)
            stats = calculate_stats(audio_output, target)

        # print(' * Test Acc {acc:.4f}'.format(acc=test_acc.avg))

        # save the prediction here
        if mode != 'all_seq':
            exp_dir = args.exp_dir
            if os.path.exists(exp_dir+'/predictions') == False:
                os.mkdir(exp_dir+'/predictions')
                np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
            np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

    return stats, loss, test_acc.avg

def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        cum_predictions = cum_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')

    cum_predictions = cum_predictions / epoch
    np.savetxt(exp_dir+'/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

    stats = calculate_stats(cum_predictions, target)
    return stats

def validate_wa(model, val_loader, args, start_epoch, end_epoch, device):
    exp_dir = args.exp_dir

    sdA = torch.load(exp_dir + '/models/model.' + str(start_epoch) + '.pth', map_location=device)

    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/model.' + str(epoch) + '.pth', map_location=device)
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1

        # if choose not to save models of epoch, remove to save space
        if args.save_model == False:
            os.remove(exp_dir + '/models/model.' + str(epoch) + '.pth')

    # averaging
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)

    model.load_state_dict(sdA)

    torch.save(model.state_dict(), exp_dir + '/models/audio_model_wa.pth')

    stats, loss = validate(model, val_loader, args, 'wa')
    return stats