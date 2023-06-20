import os
import torch
from torchvision import transforms


def get_transforms_vals(dict):

    tr_list = {}    
    tr_list["base"] = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    for k, v in dict.items():
        if k == 'rot':
            for r in v:
                tr_list["rot_"+str(r)] = transforms.Compose([
                        transforms.Resize(256, interpolation=3),
                        transforms.CenterCrop(224),
                        transforms.RandomRotation(r, interpolation=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        elif k == 'bright':
            for b in v:
                tr_list["bright_"+str(b)] = transforms.Compose([
                        transforms.Resize(256, interpolation=3),
                        transforms.CenterCrop(224),
                        transforms.ColorJitter(brightness=b),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        elif k == 'contrast':
            for c in v:
                tr_list["contrast_"+str(c)] = transforms.Compose([
                        transforms.Resize(256, interpolation=3),
                        transforms.CenterCrop(224),
                        transforms.ColorJitter(contrast=c),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        elif k == 'sat':
            for s in v:
                tr_list["sat_"+str(s)] = transforms.Compose([
                        transforms.Resize(256, interpolation=3),
                        transforms.CenterCrop(224),
                        transforms.ColorJitter(saturation=s),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        elif k == 'hue':
            for h in v:
                tr_list["hue_"+str(h)] = transforms.Compose([
                        transforms.Resize(256, interpolation=3),
                        transforms.CenterCrop(224),
                        transforms.ColorJitter(hue=h),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        elif k == 'bcsh':
            for bcsh in v:
                tr_list["bcsh_{}_{}_{}_{}".format(bcsh[0], bcsh[1], bcsh[2], bcsh[3])] = transforms.Compose([
                        transforms.Resize(256, interpolation=3),
                        transforms.CenterCrop(224),
                        transforms.ColorJitter(brightness=bcsh[0], contrast=bcsh[1], saturation=bcsh[2], hue=bcsh[3]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        elif k == 'posterize':
            for p in v:
                tr_list["posterize_".format(p)] = transforms.Compose([
                        transforms.Resize(256, interpolation=3),
                        transforms.CenterCrop(224),
                        transforms.RandomPosterize(bits=p),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
    return tr_list   

