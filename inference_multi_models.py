from faulthandler import cancel_dump_traceback_later
import rasterio
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet, EDSN, SRResNet
from basicsr.archs.rcan_arch import RCAN
from basicsr.archs.swinir_arch import SwinIR
from realesrgan import RealESRGANer
import os

def main():
    model_path = "experiments/EA-SGAN_9.28night_casaea_scharr+msLoG_edgeloss10.0/models/net_g_latest.pth"
    input_root = "data/test/LRR/"
    output_root = "results1"
    scale = 3
    tile = 0
    tile_pad = 10
    pre_pad = 10
    half = False

    # 获取输入 tif 列表
    inputs = [os.path.join(input_root, f) for f in sorted(os.listdir(input_root)) if f.lower().endswith('.tif')]
    if not inputs:
        print("没有找到tif文件，请查看目录是否为空")
        return

    for input_path in inputs:
        save_dir = os.path.join(output_root, os.path.splitext(os.path.basename(input_path))[0])
        os.makedirs(save_dir, exist_ok=True)
        inference_model(model_path, input_path, save_dir, scale=scale, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, half=half)

    print(f"共{len(inputs)}张图片推理完毕。")

def main1():
    experiment_root = "experiments"
    # input_path = "data/test/LR/tile_0_0.tif"
    input_root = "data/test/LR"
    pth_name = "net_g_latest.pth"
    output_root = "results"
    scale = 3
    tile = 0
    tile_pad = 10
    pre_pad = 10
    half = False



    # 获取输入 tif 列表
    inputs = [os.path.join(input_root, f) for f in sorted(os.listdir(input_root)) if f.lower().endswith('.tif')]
    if not inputs:
        print("没有找到tif文件，请查看目录是否为空")
        return


    for input_path in inputs:
        save_dir = os.path.join(output_root, os.path.splitext(os.path.basename(input_path))[0])
        os.makedirs(save_dir, exist_ok=True)


        # 遍历 experiments 下的所有模型子目录
        for exp_name in sorted(os.listdir(experiment_root)):
            model_dir = os.path.join(experiment_root,exp_name,"models")
            if not os.path.isdir(model_dir):
                continue

            # 尝试寻找模型权重文件
            model_path = os.path.join(model_dir,pth_name)
            if not os.path.isfile(model_path):
                print(f"在{exp_name}中，没找到权重文件")
                continue

            # 使用当前权重
            print(f"\n=== 使用模型：{model_path} ===")

            try:
                inference_model(model_path, input_path, save_dir, scale=scale, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, half=half)
            except Exception as e:
                print(f"推理失败，错误：{e}")
                continue

def inference_model(model_path, input_path, output_dir, scale=3, tile=0, tile_pad=10, pre_pad=10, half=False):
    # 1.读取影像
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        img = src.read().astype(np.float32) #[C,H,W]

    if img.shape[0] != 4:
        raise ValueError("输入影像非四波段！")

    img = img.transpose(1,2,0) #[H,W,C]
    min_val = img.min(axis=(0,1), keepdims=True)
    max_val = img.max(axis=(0,1), keepdims=True)

    img_norm = (img - min_val) / (max_val - min_val + 1e-8)

    # 2.定义模型
     # Bicubic 推理分支
    if model_path.lower() == "bicubic":
        from skimage.transform import resize
        # 使用 bicubic 插值到 HR 尺寸
        H, W, C = img.shape
        H_hr, W_hr = H * scale, W * scale
        output = resize(img, (H_hr, W_hr, C), order=3, mode='reflect', anti_aliasing=True)
        output_img = (output * (max_val - min_val) + min_val).astype(np.float32)
    else:
        # model = SwinIR(
        #     upscale=3,
        #     in_chans=4,
        #     window_size=8,
        #     img_range=1.0,
        #     depths=[6, 6, 6, 6],
        #     embed_dim=96,
        #     num_heads=[6, 6, 6, 6],
        #     mlp_ratio=4.0,
        #     qkv_bias=True,
        #     drop_path_rate=0.1,
        #     upsampler='pixelshuffle',
        #     resi_connection='1conv'
        #     )
        # model = SRResNet(
        #     num_in_ch=4,
        #     num_out_ch=4,
        #     num_feat=64,
        #     num_block=16,
        #     scale=3
        #     )
        model = RRDBNet(
            num_in_ch = 4,
            num_out_ch = 4,
            num_feat = 64,
            num_block = 23,
            num_grow_ch = 32,
            scale = scale
        )
        # model = EDSN(
        #     num_in_ch = 4,
        #     num_out_ch = 4,
        #     num_feat = 64,
        #     num_block = 16,
        #     scale = scale
        # )
        # model = RCAN(
        #     num_in_ch = 4,
        #     num_out_ch = 4,
        #     num_feat = 64,
        #     num_block = 20,
        #     num_group=10,
        #     squeeze_factor=16,
        #     upscale=3,
        #     res_scale=1,
        #     img_range=1.,
        #     rgb_mean=(0., 0., 0., 0.)
        #     )

        upsampler = RealESRGANer(
            scale = scale,
            model_path = model_path,
            model = model,
            tile = tile,
            tile_pad = tile_pad,
            pre_pad = pre_pad,
            half = half
        )

        # 3.推理
        output,_ = upsampler.enhance(img_norm, outscale = scale)

        # 4.反归一化
        output_img = output.astype(np.float32)
        output_img = output_img * (max_val - min_val) + min_val


    output_img = output_img.transpose(2, 0, 1) #[C,H,W]

    # 5.更新profile
    profile.update(
        dtype = rasterio.float32,
        count = output_img.shape[0],
        height = output_img.shape[1],
        width = output_img.shape[2]
    )
    origin_transform = profile['transform']
    new_transform = rasterio.Affine(
        origin_transform.a / scale, origin_transform.b, origin_transform.c,
        origin_transform.d, origin_transform.e / scale, origin_transform.f
    )
    profile.update(transform=new_transform)

    # 6.生成输出文件名
    input_name = os.path.basename(input_path)
    input_stem = os.path.splitext(input_name)[0]

    if model_path.lower() == "bicubic":
        output_name = f"{input_stem}_bicubic.tif"
        print_name = "Bicubic"
    else:
        exp_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        output_name = f"{input_stem}_{exp_name}.tif"
        print_name = exp_name
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    # 7.保存tif
    with rasterio.open(output_path,'w',**profile) as dst:
        dst.write(output_img)

    print(f"{input_name} 使用 {print_name} 推理完成，结果保存至{output_path}")

if __name__ == "__main__":
    main()

