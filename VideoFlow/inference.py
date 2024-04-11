
from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.utils.misc import process_cfg
from core.utils import flow_viz

from core.Networks import build_network

from core.utils import frame_utils
from core.utils.utils import InputPadder, forward_interpolate
import itertools
import imageio
#from myFunctions import sort_files_by_slice_number



def prepare_image(seq_dir):
    print(f"preparing image...")
    print(f"Input image sequence dir = {seq_dir}")

    images = []

    #image_list = sort_files_by_slice_number(os.listdir(seq_dir))
    image_list = os.listdir(seq_dir)

    for im in image_list:
        if ".jpg" in im:
            img = Image.open(seq_dir + "/" + im)
            img = np.array(img).astype(np.uint8)
            img = img[np.newaxis, :]

            #img = torch.from_numpy(img).permute(2, 0, 1).float()
            img = torch.from_numpy(img).float()
            img = torch.stack([img,img,img]).squeeze(1)
            images.append(img)
    
    return torch.stack(images)


def sort_files_by_slice_number(file_list):
    # Define a custom sorting key function
    def get_slice_number(filename):
        if '_D_' in filename:
            slice_number_str = filename.split('_')[5]
        else:
            slice_number_str = filename.split('_')[4]
        # Convert the extracted string to an integer
        return int(slice_number_str)

    # Use the sorted function with the custom key function
    sorted_files = sorted(file_list, key=get_slice_number)

    return sorted_files

'''
def prepare_image(seq_dir):
    print(f"preparing image...")
    print(f"Input image sequence dir = {seq_dir}")

    images = []

    image_list = sorted(os.listdir(seq_dir))

    for fn in image_list:
        img = Image.open(os.path.join(seq_dir, fn))
        img = np.array(img).astype(np.uint8)[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        images.append(img)
    
    return torch.stack(images)
'''



def vis_pre(flow_pre, vis_dir):

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    N = flow_pre.shape[0]

    for idx in range(N//2):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx+2, idx+3))
    
    for idx in range(N//2, N):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx-N//2+2, idx-N//2+1))



@torch.no_grad()
def MOF_inference(model, cfg):

    model.eval()

    input_images = prepare_image(cfg.seq_dir)
    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(input_images.shape)

    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction
    
    flow_pre, _ = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()

    return flow_pre

@torch.no_grad()
def BOF_inference(model, cfg):

    model.eval()

    input_images = prepare_image(cfg.seq_dir)
    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)
    flow_pre, _ = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()

    return flow_pre

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='MOF')
    parser.add_argument('--seq_dir', default='C:/Users/loren/Documents/Data/BlastoData/blasto/D2013.02.19_S0675_I141_1')
    parser.add_argument('--vis_dir', default='C:/Users/loren/Documents/Data/BlastoData/opticalFlowFrames/VideoFlow/blasto/D2013.02.19_S0675_I141_1')

    args = parser.parse_args()

    if args.mode == 'MOF':
        from configs.multiframes_sintel_submission import get_cfg
    elif args.mode == 'BOF':
        from configs.sintel_submission import get_cfg

    # Importo le configurazioni relative alla rete di interesse e poi le salvo in cfg.
    # Le cfg sono molte e sono quelle necessarie per costruire il modello. Tra queste vi sono le due cartelle, quella
    # da cui prendere i dati e quella in cui salvare le immagini del flusso ottico (passate da terminale o di default)  
    cfg = get_cfg()
    cfg.update(vars(args))

    # Il modello viene costruito con il build_network e viene parallelizzato grazie al DataParallel di pytorch
    # Il build_network è importato da core.Networks e, basandosi sulle cfg, contiene le info per la costruzione del network
    model = torch.nn.DataParallel(build_network(cfg))
    # Poi, dato che uso modelli preaddestrati, carico pesi e modello della configurazione di interesse
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()    # Giro su GPU
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))

    print("===============================================================")
    print(cfg)
    print("===============================================================")
    
    breakpoint()

    with torch.no_grad():
        if args.mode == 'MOF':
            from configs.multiframes_sintel_submission import get_cfg
            flow_pre = MOF_inference(model.module, cfg)
        elif args.mode == 'BOF':
            from configs.sintel_submission import get_cfg
            flow_pre = BOF_inference(model.module, cfg)
    
    vis_pre(flow_pre, cfg.vis_dir)


# runno con:
# python -u inference.py --mode MOF --seq_dir demo_input_images --vis_dir demo_flow_vis

'''
La "-u" sta per "unbuffered", ovvero che gli output vengono subito mostrati senza aspettare che termini lo script
"if __name__ == '__main__':" --> quando lo script viene richiamato da terminale con "python script_name.py" questo assume il
name di "__main__" e quindi verrà rispettata la condizione. Verrà parsata la stringa e  

'''




