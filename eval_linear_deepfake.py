# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
import validation_transforms

import utils
import vision_transformer as vits
import visdom as vis
import webdataset as wds
from torchvision.utils import save_image


def get_dataset_len(dataset):
    i = 0
    for x, l in dataset:
        i += 1
    return i


def wds_deepfake_generator_jpg(dataset_instance):
    i = 0
    for sample in dataset_instance:
        del sample['json']
        del sample['fake_1.jpg']
        del sample['fake_2.jpg']
        del sample['fake_3.jpg']
        del sample['fake_4.jpg']
        if i % 2 == 0:
            del sample['fake_0.jpg']
            sample['cls'] = torch.tensor([0], dtype=torch.float32)
        else:
            sample['jpg'] = sample['fake_0.jpg']
            del sample['fake_0.jpg']
            sample['cls'] = torch.tensor([1], dtype=torch.float32)
        i += 1
        yield sample

def wds_deepfake_generator_png(dataset_instance):
    i = 0
    for sample in dataset_instance:
        del sample['json']
        del sample['fake_1.jpg']
        del sample['fake_2.jpg']
        del sample['fake_3.jpg']
        del sample['fake_4.jpg']
        if i % 2 == 0:
            del sample['fake_0.jpg']
            sample['cls'] = torch.tensor([0], dtype=torch.float32)
        else:
            sample['jpg'] = sample['fake_0.jpg']
            del sample['fake_0.jpg']
            sample['cls'] = torch.tensor([1], dtype=torch.float32)
        i += 1
        yield sample

def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    global url_val
    global url_test
    global dataset_val
    global val_loader

    if args.format == "jpg":
        url_val = "/work/tesi_tpoppi/deepfake_1/coco-384-validation-dict-625-{000..007}.tar"
        url_test = "/work/tesi_tpoppi/deepfake_1/coco-384-test-dict-625-{000..007}.tar"
        if args.debug:
            url_val = "/work/tesi_tpoppi/deepfake_1/coco-384-validation-dict-625-000.tar"
            url_test = "/work/tesi_tpoppi/deepfake_1/coco-384-test-dict-625-000.tar"
        
        batch_size = args.batch_size_per_gpu #defined in eval_linear script

        dataset_val = (wds.WebDataset(url_test).decode('pil')
                   .compose(wds_deepfake_generator_jpg)
                   .to_tuple('jpg', 'cls')
                   .map_tuple(val_transform, lambda x:x)
                   .shuffle(1000))

        dataset_val = dataset_val.with_length(get_dataset_len(dataset_val))

        val_loader = wds.WebLoader(
                dataset_val, batch_size=batch_size, shuffle=False, num_workers=2,
        )

        val_loader = val_loader.with_length(len(dataset_val))

    if args.format == "png":
        png_dataset_path = "/work/tesi_tpoppi/dataset_png"
        dataset_val = datasets.ImageFolder(os.path.join(png_dataset_path, "test"), transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )


    if args.evaluate:
        utils.load_custom_linear_weights(linear_classifier, os.path.join(args.output_dir, "checkpoint.pth.tar"))
        test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if args.transforms:
            #dict_transforms = {"rot": [5, 10, 30],
            #                   "bright": [.2, .5, .8],
            #                   "contrast": [.2, .5, .8],
            #                   "sat": [.2, .5, .8],
            #                   "hue": [.2],
            #                   "bcsh": [[.2,.2,.2,0], [.5,.5,0,0], [.5,.5,.2,.2]],
            #                   "posterize": [6, 4, 2]}
            dict_transforms = {"posterize": [6, 4, 2]}
            transformations_list = validation_transforms.get_transforms_vals(dict_transforms)

            for k, v in transformations_list.items():
                transform_dataset_val = (wds.WebDataset(url_test).decode('pil')
                                    .compose(wds_deepfake_generator_jpg)
                                    .to_tuple('jpg', 'cls')
                                    .map_tuple(v, lambda x:x)
                                    .shuffle(1000))
                
                transform_dataset_val = transform_dataset_val.with_length(len(dataset_val))

                transform_val_loader = wds.WebLoader(transform_dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)
                transform_val_loader = transform_val_loader.with_length(len(dataset_val))
                
                test_stats = validate_network(transform_val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, True, k, args.output_dir)
                print(f"Accuracy of the network on the {k} validation transform: {test_stats['acc1']:.1f}%")

        return

    global dataset_train
    global train_loader
    
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.format == "jpg":
        dataset_train = (wds.WebDataset(url_val).decode('pil')
                    .compose(wds_deepfake_generator_jpg)
                    .to_tuple('jpg', 'cls')
                    .map_tuple(train_transform, lambda x:x)
                    .shuffle(1000, initial=1000))
        dataset_train = dataset_train.with_length(get_dataset_len(dataset_train))

        train_loader = wds.WebLoader(
                dataset_train, batch_size=batch_size, shuffle=False, num_workers=2,)
        train_loader = train_loader.with_length(len(dataset_train))

    elif args.format == "png":
        dataset_train = datasets.ImageFolder(os.path.join(png_dataset_path, "train"), transform=train_transform)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
    


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)

        # compute binary cross entropy loss
        loss = nn.BCELoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool, save_imgs = False, transf_type=None, out_dir=None):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    c = 0
    im_dir = ""

    if save_imgs:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        im_dir = os.path.join(out_dir, "imgs")
        if not os.path.exists(im_dir):
            os.mkdir(im_dir)

    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
                
        output = linear_classifier(output)
        
        if save_imgs==True:
            #save 1 image for each batch size
            im = inp[17, :, :, :]
            save_image(im, os.path.join(im_dir, "img_{}__{}.png".format(transf_type, str(c))))
            with open(os.path.join(im_dir, 'img_{}__{}_target.txt'.format(transf_type, str(c))), 'w') as f:
                f.write(str(target[17,:]))
            with open(os.path.join(im_dir, 'img_{}__{}_pred.txt'.format(transf_type, str(c))), 'w') as f:
                f.write(str(output[17,:]))
            c += 1
                    
        loss = nn.BCELoss()(output, target)

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.get_accuracy(output, target)

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=2):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)

    def forward(self, x):
        out = self.linear(x)
        out = nn.functional.sigmoid(out)

        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with binary linear classification for deepfake images.')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=105, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--debug', dest='debug', action='store_true', help='run in debug mode (smaller dataset and faster)')
    parser.add_argument('--transforms_pipeline', dest='transforms', action='store_true', help="""set up an evaluation
        pipeline which tests validation set on several different transformations.""")
    parser.add_argument('--dataset_format', dest='format', default='jpg')
    args = parser.parse_args()
    eval_linear(args)
