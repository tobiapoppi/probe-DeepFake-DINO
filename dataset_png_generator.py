import pathlib
import shutil
import os

png_folder_fake = "/mnt/beegfs/work/publicfiles/drive/elsa_dataset/version_1/media_analytics_challenge/ELSA/dataset/fake-images"
png_folder_real = "/work/tesi_tpoppi/laion_real_fake_new"

real = pathlib.Path(png_folder_real)
fake = pathlib.Path(png_folder_fake)

ds_fake = fake.rglob("*.png")
ds_real = real.rglob("*.png")

for i in range(40000):
    im = next(ds_fake)
    shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/train/1", im.name))

for i in range(10000):
    im = next(ds_fake)
    shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/test/1", im.name))

for i in range(40000):
    im = next(ds_real)
    shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/train/0", im.name))

for i in range(10000):
    im = next(ds_real)
    shutil.copyfile(im, os.path.join("/work/tesi_tpoppi/dataset_png/test/0", im.name))