import argparse
import glob
import mimetypes
import os
import pathlib
import shutil
import subprocess
import tempfile
from collections import OrderedDict

import cv2
import magic
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from main_test_swinir import define_model, get_image_pair, setup


class Predictor(BasePredictor):
    def setup(self):
        model_dir = '/experiments/pretrained_models'

        self.model_zoo = {
            'real_sr': {
                4: os.path.join(model_dir, '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth')
            },
            'gray_dn': {
                15: os.path.join(model_dir, '004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth'),
                25: os.path.join(model_dir, '004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth'),
                50: os.path.join(model_dir, '004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth')
            },
            'color_dn': {
                15: os.path.join(model_dir, '005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth'),
                25: os.path.join(model_dir, '005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth'),
                50: os.path.join(model_dir, '005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth')
            },
            'jpeg_car': {
                10: os.path.join(model_dir, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth'),
                20: os.path.join(model_dir, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth'),
                30: os.path.join(model_dir, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth'),
                40: os.path.join(model_dir, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth')
            }
        }

        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, default='real_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                        'gray_dn, color_dn, jpeg_car')
        parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
        parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
        parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
        parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                                                                 'Just used to differentiate two different settings in Table 2 of the paper. '
                                                                                 'Images are NOT tested patch by patch.')
        parser.add_argument('--large_model', action='store_true',
                            help='use large model, only provided for real image sr')
        parser.add_argument('--model_path', type=str,
                            default=self.model_zoo['real_sr'][4])
        parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
        parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')

        self.args = parser.parse_args('')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tasks = {
            'Real-World Image Super-Resolution': 'real_sr',
            'Grayscale Image Denoising': 'gray_dn',
            'Color Image Denoising': 'color_dn',
            'JPEG Compression Artifact Reduction': 'jpeg_car'
        }
        self.model = None


    def predict(self, 
        image: Path = Input(description="input image"),
        task_type: str = Input(
            description='image restoration task type',
            default='Real-World Image Super-Resolution',
            choices=['Real-World Image Super-Resolution', 'Grayscale Image Denoising', 'Color Image Denoising','JPEG Compression Artifact Reduction']        
        ),  
        jpeg: int = Input(
            description="scale factor, activated for JPEG Compression Artifact Reduction. ",
            default=40),
        noise: int = Input(
             description="noise level, activated for Grayscale Image Denoising and Color Image Denoising.",
             default=15)
        ) -> Path:
        self.args.task = self.tasks[task_type]
        self.args.noise = noise
        self.args.jpeg = jpeg

        # set model path
        if self.args.task == 'real_sr':
            self.args.scale = 4
            self.args.model_path = self.model_zoo[self.args.task][4]
        elif self.args.task in ['gray_dn', 'color_dn']:
            self.args.model_path = self.model_zoo[self.args.task][noise]
        else:
            self.args.model_path = self.model_zoo[self.args.task][jpeg]

        mimestart = magic.from_file(image, mime=True)
        if mimestart is None:
            raise Exception("Could not determine file type of " + str(image))
        mimestart = mimestart.split('/')[0]
        is_video = mimestart == 'video'
        # print("is_video", is_video)

        if is_video:
            # Save video framerate
            # Execute  !ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$image"
            # using python subprocess
            args = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', str(image)]
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            framerate = out.decode('utf-8').strip()
            print("framerate", framerate)
            multiplier, divisor = framerate.split('/')
            framerate = float(multiplier) / float(divisor)
            print("framerate", framerate)

            # Extract frames
            # execute !ffmpeg -i "$video_file" /%05d.png
            # using python os.system
            frames_path = Path('/output/')
            frames_path.mkdir(exist_ok=True)
            os.system('ffmpeg -i {} {}/%05d.png'.format(image, frames_path))

            # Process frames
            frames = list(frames_path.glob('*.png'))
            frames.sort()
            for frame in frames:
                print("upscaling path", frame)
                path = self.predict(frame, task_type, jpeg, noise)
                print("upscaled path", path)
                os.system('mv -v {} {}'.format(path, frame))
                # anything in the /outputs folder will be shown as feedback to the user in the pollinations UI
                os.system('cp -v {} {}'.format(frame, "/outputs/a_progress.png"))

            # Create video
            # execute !ffmpeg -framerate "$framerate" -i /%05d  -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "$output_file"
            # using python os.system
            output_file = Path('/output/output.mp4')
            os.system('ffmpeg -framerate {} -i {}/%05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {}'.format(framerate, frames_path, output_file))
            return output_file
        try:
            
            # set input folder
            input_dir = 'input_cog_temp'
            os.makedirs(input_dir, exist_ok=True)
            input_path = os.path.join(input_dir, os.path.basename(image))
            shutil.copy(str(image), input_path)
            if self.args.task == 'real_sr':
                self.args.folder_lq = input_dir
            else:
                self.args.folder_gt = input_dir
            # print("loading model", self.args.model_path)
            if not self.model:
                self.model = define_model(self.args)
                self.model.eval()
                self.model = self.model.to(self.device)
            # print("loaded model")
            # setup folder and path
            folder, save_dir, border, window_size = setup(self.args)
            os.makedirs(save_dir, exist_ok=True)
            test_results = OrderedDict()
            test_results['psnr'] = []
            test_results['ssim'] = []
            test_results['psnr_y'] = []
            test_results['ssim_y'] = []
            test_results['psnr_b'] = []
            # psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0
            out_path = pathlib.Path(tempfile.mkdtemp()) / "out.png"

            for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
                # read image
                imgname, img_lq, img_gt = get_image_pair(self.args, path)  # image to HWC-BGR, float32
                img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                                      (2, 0, 1))  # HCW-BGR to CHW-RGB
                img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

                # inference
                with torch.no_grad():
                    # pad input image to be a multiple of window_size
                    _, _, h_old, w_old = img_lq.size()
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                    output = self.model(img_lq)
                    output = output[..., :h_old * self.args.scale, :w_old * self.args.scale]

                # save image
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if output.ndim == 3:
                    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
                cv2.imwrite(str(out_path), output)
        finally:
            clean_folder(input_dir)
        return Path(out_path)


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
