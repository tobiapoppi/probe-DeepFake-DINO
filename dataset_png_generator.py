import pathlib
import shutil
import os
from PIL import Image

png_folder_fake = "/mnt/beegfs/work/publicfiles/drive/elsa_dataset/version_1/media_analytics_challenge/ELSA/dataset/fake-images"
png_folder_real = "/work/tesi_tpoppi/laion_real_fake_new"

real = pathlib.Path(png_folder_real)
fake = pathlib.Path(png_folder_fake)

ds_fake = fake.rglob("*.png")
ds_real = real.rglob("*.png")

def adj_names():
    count = 0
    f1 = pathlib.Path("/work/tesi_tpoppi/dataset_png/train/1")
    f2 = pathlib.Path("/work/tesi_tpoppi/dataset_png/test/1")
    ims1 = f1.rglob("*.png")
    for im in ims1:
        count += 1
        os.rename(str(im), os.path.join(pathlib.Path(str(im)).parent, str(count) + ".png"))
    
    ims2 = f2.rglob("*.png")
    count = 0
    for im in ims2:
        count += 1
        os.rename(str(im), os.path.join(pathlib.Path(str(im)).parent, str(count) + ".png"))


def check(im):
    try:
        Image.open(im)
        return True
    except:
        return False

def create():
    ## dati i lunghi tempi di download per laion, cercando di tenere il dataset equilibrato teniamo:
    ## 57600 img di train
    ## 14400 img di test
    
    count = 0
    while count < 28800:
        im = next(ds_fake)
        if check(im):
            shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/train/1", str(count) + ".png"))
            count += 1
    count = 0
    while count < 7200:
        im = next(ds_fake)
        if check(im):
            shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/test/1", str(count) + ".png"))
            count += 1
    count = 0
    while count < 28700:
        im = next(ds_real)
        if check(im):
            shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/train/0", str(count) + ".png"))
            count += 1
    count = 0
    while count < 7100:
        im = next(ds_real)
        if check(im):
            shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/test/0", str(count) + ".png"))
            count += 1

def create_debug():
    
    count = 0
    while count < 1000:
        im = next(ds_fake)
        if check(im):
            shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/debug/train/1", str(count) + ".png"))
            count += 1
    count = 0
    while count < 200:
        im = next(ds_fake)
        if check(im):
            shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/debug/test/1", str(count) + ".png"))
            count += 1
    count = 0
    while count < 1000:
        im = next(ds_real)
        if check(im):
            shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/debug/train/0", str(count) + ".png"))
            count += 1
    count = 0
    while count < 200:
        im = next(ds_real)
        if check(im):
            shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/debug/test/0", str(count) + ".png"))
            count += 1


create()
#create_debug()