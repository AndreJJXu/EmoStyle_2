import os

from argparse import Namespace
import time
import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision

sys.path.append(".")
sys.path.append("..")

from utils_latent.common import tensor2im
from models_latent.psp import pSp

def Inference(imagePath, classtype):
    print(imagePath, classtype)
    experiment_type = classtype
    # experiment_type = 'church_encode' #@param ['ffhq_encode', 'cars_encode', 'horse_encode', 'church_encode']

    MODEL_PATHS = {
        "ffhq_encode": {"id": "1cUv_reLE6k3604or78EranS7XzuVMWeO", "name": "e4e_ffhq_encode.pt"},
        "cars_encode": {"id": "17faPqBce2m1AQeLCLHUVXaDfxMRU2QcV", "name": "e4e_cars_encode.pt"},
        "horse_encode": {"id": "1TkLLnuX86B_BMo2ocYD0kX9kWh53rUVX", "name": "e4e_horse_encode.pt"},
        "church_encode": {"id": "1-L0ZdnQLwtdy6-A_Ccgq5uNJGTqE7qBa", "name": "e4e_church_encode.pt"}
    }

    # path = MODEL_PATHS[experiment_type]
    # downloader.download_file(file_id=path["id"], file_name=path["name"])

    EXPERIMENT_DATA_ARGS = {
        "ffhq_encode": {
            "model_path": "optimization/pretrained_models/e4e_ffhq_encode.pt",
            "image_path": "notebooks/images/input_img.jpg"
        },
        "cars_encode": {
            "model_path": "optimization/pretrained_models/e4e_cars_encode.pt",
            "image_path": "notebooks/images/car_img.jpg"
        },
        "horse_encode": {
            "model_path": "optimization/pretrained_models/e4e_horse_encode.pt",
            "image_path": "notebooks/images/horse_img.jpg"
        },
        "church_encode": {
            "model_path": "optimization/pretrained_models/e4e_church_encode.pt",
            "image_path": "notebooks/images/church_img.jpg"
        }
        
    }

    # Setup required image transformations
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]
    if experiment_type == 'cars_encode':
        EXPERIMENT_ARGS['transform'] = transforms.Compose([
                transforms.Resize((192, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        resize_dims = (256, 192)
    else:
        EXPERIMENT_ARGS['transform'] = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        resize_dims = (256, 256)

    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    # pprint.pprint(opts)  # Display full options used
    # update the training options
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')
    image_path = imagePath
    # image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")

    if experiment_type == "ffhq_encode" and 'shape_predictor_68_face_landmarks.dat' not in os.listdir():
        print("Downloading!!!!!")

    def run_alignment(image_path):
        import dlib
        from utils_latent.alignment import align_face
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        aligned_image = align_face(filepath=image_path, predictor=predictor) 
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image 

    if experiment_type == "ffhq_encode":
        input_image = run_alignment(image_path)
    else:
        input_image = original_image

    input_image.resize(resize_dims)

    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)

    def display_alongside_source_image(result_image, source_image):
        res = np.concatenate([np.array(source_image.resize(resize_dims)),
                            np.array(result_image.resize(resize_dims))], axis=1)
        
        return Image.fromarray(res)

    def run_on_batch(inputs, net):
        images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
        print(images.shape)
        if experiment_type == 'cars_encode':
            images = images[:, :, 32:224, :]
        return images, latents

    with torch.no_grad():
        tic = time.time()
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
        result_image, latent = images[0], latents[0]
        print(result_image.shape, latent.shape, "!!!!!!!!!!!!!!!!!!!!!!")
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    # Display inversion:
    im = display_alongside_source_image(tensor2im(result_image), input_image)
    im.save("resultszzh.jpg")
    # print(latents)
    latents = torch.squeeze(latents)
    torch.save(latents, 'latent_codezzh.pt')
    print(latents.shape, '-------')
    return latents


if __name__ == '__main__':
    imagePath = r"optimization/notebooks/images/input_img.jpg"
    classtype = "church_encode"
    Inference(imagePath, classtype)