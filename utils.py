import torch
from torchvision import transforms
from PIL import Image
from models import SuperResolution, Segmentation

def load_sr_model(checkpoint_path, device='cuda'):
    model = SuperResolution(
        img_size=64,
        patch_size=1,
        in_chans=1,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2.0,
        upsampler='pixelshuffle',
        resi_connection='1conv',
        upscale=4,
        img_range=1.0
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def prepare_image(img_path, device):
    img = Image.open(img_path).convert('L')
    to_tensor = transforms.ToTensor()
    inp = to_tensor(img).unsqueeze(0).to(device)
    return inp

def load_segmentation_model(checkpoint_path, task="segment"):
    return Segmentation(checkpoint_path, task=task)

