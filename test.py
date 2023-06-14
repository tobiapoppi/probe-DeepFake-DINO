import torch
from torchvision import datasets
import webdataset as wds
from itertools import islice
from torchvision import transforms

url_val = "/work/tesi_tpoppi/deepfake_1/coco-384-validation-dict-625-{000..007}.tar"
url_test = "/work/tesi_tpoppi/deepfake_1/coco-384-test-dict-625-{000..007}.tar"

#    dataset_instance = wds.WebDataset(path).decode('pil').to_tuple('jpg')\

batch_size = 199999 #defined in eval_linear script

def get_dataset_len(dataset):
    i = 0
    for x, l in dataset:
        i += 1
    return i

def wds_deepfake_generator(dataset_instance):
    i = 0
    for sample in dataset_instance:
        print(type(sample))
        del sample['json']
        del sample['fake_1.jpg']
        del sample['fake_2.jpg']
        del sample['fake_3.jpg']
        del sample['fake_4.jpg']
        if i % 2 == 0:
            del sample['fake_0.jpg']
            sample['cls'] = "0"
        else:
            sample['jpg'] = sample['fake_0.jpg']
            del sample['fake_0.jpg']
            sample['cls'] = "1"
        i += 1
        yield sample

dataset_val = (wds.WebDataset(url_val).decode('pil')
               .compose(wds_deepfake_generator)
               .to_tuple('jpg', 'cls')
               .map_tuple(transforms.ToTensor(), lambda x:x)
               .map_tuple(transforms.Resize((384, 384)), lambda x:x)
               .shuffle(100)
               .batched(batch_size))

val_loader = wds.WebLoader(
        dataset_val, batch_size=None, shuffle=False, num_workers=4,
    )

batch = next(iter(val_loader))
print(batch[0].shape, batch[1].shape)


print('dataset caricato... decoding e tensorizzazione in corso...\n')




#sample è un oggetto identificato dal nome di un'immagine
#ogni sample è una lista di chiavi valori

#val_loader = wds.WebLoader(url).decode( wds.handle_extension("left.png", png_decoder_16bpp),
#    wds.handle_extension("right.png", png_decoder_16bpp),
#    wds.imagehandler("torchrgb")
#)