import h5py
import torch
import tokyymodel as tm
import numpy as np
import matplotlib.pyplot as plt

test_what = input("Test what? [ show | batch ]")

if test_what == "show":
    file_path = r"nyu\train\kitchen_0035b\00006.h5"

    with h5py.File(file_path, 'r') as f:
        rgb = np.array(f['rgb'])
        depth = np.array(f['depth'])

    print(f"RGB min: {rgb.min()}, max: {rgb.max()}")
    print(f"Depth min: {depth.min()}, max: {depth.max()}")

    rgb_img = np.transpose(rgb, (1, 2, 0))

    plt.subplot(1, 2, 1)
    plt.title("RGB Image")
    plt.imshow(rgb_img.astype(np.uint8))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Depth Map")
    plt.imshow(depth, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.axis('off')

    plt.show()

elif test_what == "batch":
    batch_size = 16
    while True:
        try:
            dummy = torch.randn(batch_size, 3, 128, 128).cuda()
            model = tm.ResUNet().cuda()
            out = model(dummy)
            print(f"Success at batch size: {batch_size}")
            batch_size += 16
        except RuntimeError as e:
            print(f"OOM at batch size: {batch_size}")
            break