from others.params import *
from deals.utils import *
from deals import utils
import time
from torch import nn, optim
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from others.record import *
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def detect_accuary(output, target):
    pass


def train(model, train_dataloader, epoch, criterion, optimizer, writer, device, record):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    model.cuda()
    end = time.time()
    for step, inputs in enumerate(train_dataloader):
        data_time.update(time.time() - end)
        if args.dataset == 'EgoGesture' or args.dataset == 'nvGesture':
            rgb, depth, labels = inputs[0], inputs[1], inputs[2]
            rgb = rgb.to(device, non_blocking=True).float()
            # depth = depth.to(device, non_blocking=True).float()
            outputs = model(rgb)
        else:
            rgb, labels = inputs[0], inputs[1]
            rgb = rgb.to(device, non_blocking=True).float()

            rgb = rgb.cuda()

            outputs = model(rgb)
        labels = labels.to(device, non_blocking=True).long()
        loss = criterion(outputs, labels)
        # measure accuracy and record loss
        if args.is_detector_classify == "classify":
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 1))
        losses.update(loss.item(), labels.size(0))
        top1.update(prec1.item(), labels.size(0))
        top5.update(prec5.item(), labels.size(0))

        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (step + 1) % params['display'] == 0:
            print_string = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}, '
                            'data_time: {data_time.val:.3f} ({data_time.avg:.3f}), batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                            'loss: {loss.val:.4f} ({loss.avg:.4f}), '
                            'Top-1: {top1_acc.val:.2f} ({top1_acc.avg:.2f}), '
                            'Top-5: {top5_acc.val:.2f} ({top5_acc.avg:.2f})'
                            .format(epoch, step + 1, len(train_dataloader),
                                    lr=optimizer.param_groups[2]['lr'],
                                    data_time=data_time, batch_time=batch_time,
                                    loss=losses, top1_acc=top1, top5_acc=top5
                                    )
                            )
            print(print_string)
    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('train_top5_acc_epoch', top5.avg, epoch)
    record.train_loss.append(losses.avg)
    record.train_acc_top1.append(top1.avg)
    record.train_acc_top5.append(top5.avg)


def validation(model, val_dataloader, epoch, criterion, optimizer, writer, device, record):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, inputs in enumerate(val_dataloader):
            data_time.update(time.time() - end)

            if args.dataset == 'EgoGesture' or args.dataset == 'nvGesture':
                rgb, depth, labels = inputs[0], inputs[1], inputs[2]
                rgb = rgb.to(device, non_blocking=True).float()
                outputs = model(rgb)
            else:
                rgb, labels = inputs[0], inputs[1]
                rgb = rgb.to(device, non_blocking=True).float()
                outputs = model(rgb)
            labels = labels.to(device, non_blocking=True).long()

            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            if args.is_detector_classify == "classify":
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            else:
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 1))
            losses.update(loss.item(), labels.size(0))
            top1.update(prec1.item(), labels.size(0))
            top5.update(prec5.item(), labels.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if (step + 1) % params['display'] == 0:
                print_string = ('Test: [{0}][{1}], '
                                'data_time: {data_time.val:.3f} ({data_time.avg:.3f}), batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                                'loss: {loss.val:.4f} ({loss.avg:.4f}), '
                                'Top-1: {top1_acc.val:.2f} ({top1_acc.avg:.2f}), '
                                'Top-5: {top5_acc.val:.2f} ({top5_acc.avg:.2f})'
                                .format(step + 1, len(val_dataloader),
                                        data_time=data_time, batch_time=batch_time,
                                        loss=losses, top1_acc=top1, top5_acc=top5
                                        )
                                )
                print(print_string)
        print_string = ('Testing Results: loss {loss.avg:.5f}, Top-1 {top1.avg:.3f}, Top-5 {top5.avg:.3f}'
                        .format(loss=losses, top1=top1, top5=top5)
                        )
        print(print_string)

    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)
    record.val_loss.append(losses.avg)
    record.val_acc_top1.append(top1.avg)
    record.val_acc_top5.append(top5.avg)

    model.train()
    return losses.avg, top1.avg, top5.avg


def testing(model, val_dataloader, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(val_dataloader)):
            if args.dataset == 'EgoGesture' or args.dataset == 'nvGesture':
                rgb, depth, labels = inputs[0], inputs[1], inputs[2]
                rgb = rgb.to(device, non_blocking=True).float()
                # depth = depth.to(device, non_blocking=True).float()
                if args.use_video_swin_transformer == True:
                    rgb = rgb.permute(0, 2, 1, 3, 4)
                    # depth = depth.permute(0, 2, 1, 3, 4)
                outputs = model(rgb)
            else:
                rgb, labels = inputs[0], inputs[1]
                rgb = rgb.to(device, non_blocking=True).float()
                outputs = model(rgb)

            labels = labels.to(device, non_blocking=True).long()
            loss = criterion(outputs, labels)
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), labels.size(0))
            top1.update(prec1.item(), labels.size(0))
            top5.update(prec5.item(), labels.size(0))
            batch_time.update(time.time() - end)

    print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
    print(print_string)
    print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
        top1_acc=top1.avg,
        top5_acc=top5.avg)
    print(print_string)


def my_train(model, train_dataloader, val_dataloader, device, record, policies=None):
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    best_acc = 0.
    best_acc_top5 = 0.

    criterion = nn.CrossEntropyLoss().to(device)
    if policies == None:
        policies = model.get_optim_policies()
    # model = nn.DataParallel(model)  # multi-Gpu
    model = model.to(device)
    model.cuda()
    if params['recover_from_checkpoint'] is not None:
        testing(model, val_dataloader, criterion)

    for param_group in policies:
        param_group['lr'] = args.lr * param_group['lr_mult']
        param_group['weight_decay'] = args.weight_decay * param_group['decay_mult']
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    optimizer = optim.SGD(policies, momentum=params['momentum'])

    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    model_save_dir = os.path.join(params['save_path'], cur_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    for epoch in trange(params['epoch_num']):
        train(model, train_dataloader, epoch, criterion, optimizer, writer, device, record)
        if epoch % 1 == 0:
            val_loss, val_acc, top5_acc = validation(model, val_dataloader, epoch, criterion, optimizer, writer, device, record)
            if val_acc > best_acc:
                checkpoint = os.path.join(model_save_dir,
                                          "clip_len_" + str(params['clip_len']) + "frame_sample_rate_" +
                                          str(params['frame_sample_rate']) + "_checkpoint" + ".pth.tar")
                utils.save_checkpoint(model, optimizer, checkpoint)
                best_acc = val_acc

                # 获取当前文件路径
                current_file = os.path.abspath(__file__)
                # 获取所在目录路径
                current_directory = os.path.dirname(current_file)
                # 获取上一级目录路径
                parent_directory = os.path.dirname(current_directory)

                save_path = os.path.join(parent_directory + '/weights', args.base_model + args.is_detector_classify + args.describe + ".pth")
                utils.save_checkpoint(model, optimizer, save_path + '.tar')
                torch.save(model, save_path)

                # save_confusion_matrix(model, val_dataloader)

            if top5_acc > best_acc_top5:
                best_acc_top5 = top5_acc
            print('Best Top-1: {:.2f}'.format(best_acc))
            print('Best Top-5: {:.2f}'.format(best_acc_top5))

        utils.adjust_learning_rate(params['learning_rate'], optimizer, epoch, args.lr_steps)
    writer.close


def select_label(labels):
    if labels in args.system_label:
        return True
    return False


# 绘制混淆矩阵
def save_confusion_matrix(model, val_dataloader):
    model.eval()

    # 存储预测结果
    predictions = []
    true_labels = []

    # 对测试数据进行预测
    with torch.no_grad():
        for step, inputs in enumerate(val_dataloader):
            rgb, depth, labels = inputs[0], inputs[1], inputs[2]
            outputs = model(rgb)
            _, predicted = torch.max(outputs.data, 1)

            pre = predicted.cpu().numpy()
            lab = labels.cpu().numpy()
            
            # for i in range(len(lab)):
            #     if select_label(lab[i]):
            #         predictions.append(lab[i])
            #         true_labels.append(pre[i])
            #     else:
            #         continue
            predictions.extend(pre)
            true_labels.extend(lab)

    # 生成混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    cm_selected_columns = cm[args.system_label, :][:, args.system_label]

    # 设置分类的标签
    class_labels = args.system_label

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_selected_columns, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels, linewidths=.5)
    plt.xlabel("True labels")
    plt.ylabel("Predict labels")
    plt.title("The confusion matrix of partially similar gestures for IMR-Net on " + args.dataset)
    plt.grid(False)

    # 保存为图片
    plt.savefig('./confusion_matrix.png')
    # 显示图形
    plt.show()
