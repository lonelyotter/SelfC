import os.path as osp
import os
import logging
import argparse
import options.options as option
import utils.util as util
import torch
import numpy as np
from data import create_dataset, create_dataloader
from data.util import read_img1, channel_convert
from models import create_model

#### options
parser = argparse.ArgumentParser()

parser.add_argument('-opt',
                    type=str,
                    required=True,
                    help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs((path for key, path in opt['path'].items()
             if not key == 'experiments_root' and 'pretrain_model' not in key
             and 'resume' not in key))

util.setup_logger('base',
                  opt['path']['log'],
                  'test_' + opt['name'],
                  level=logging.INFO,
                  screen=True,
                  tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(
        dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)

img_ext = ".jpg"

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)
    print("dataset_dir: ", dataset_dir)
    idx = 0
    for data in test_loader:
        idx += 1
        model.feed_data(data)

        # size: (bt, 3, h, w)
        forw_L = model.test_downscale()

        bt, c, h, w = forw_L.shape
        t = 7
        b = bt // t
        lr_vis = forw_L.reshape(b, t, c, h, w)

        # save low resolution images
        video_dir = osp.join(dataset_dir, str(idx))
        util.mkdir(video_dir)
        for t_i in range(t):
            lr_1im = lr_vis[0, t_i]
            lr_1im = util.tensor2img(lr_1im)
            save_path = osp.join(video_dir, str(t_i + 1) + img_ext)
            util.save_img(lr_1im, save_path)
        
        # h264 encode
        os.system("cd " + video_dir + " && ffmpeg -f image2 -i %d.jpg -crf 10 output.mp4")

        # h264 decode
        os.system("cd " + video_dir + " && ffmpeg -i output.mp4 de_%d.jpg")

        decoded_paths = [osp.join(video_dir, str(t_i + 1) + img_ext) for t_i in range(t)]
        logger.info(decoded_paths)
        imgs = []
        for decoded_path in decoded_paths:
            img = read_img1(env=None, path=decoded_path)
            img = img[:, :, [2, 1, 0]]
            img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
            imgs.append(img)
            
        # bt, 3, h, w
        lr_video = torch.stack(imgs, dim=0)

        lr_video = lr_video.to(forw_L.device)

        logger.info(torch.mean(torch.abs(lr_video - forw_L)))

        HR_Video = model.test_upscale(lr_video)
        bt, c, h, w = HR_Video.shape
        hr_vis = HR_Video.reshape(b, t, c, h, w)
        for t_i in range(t):
            hr_1im = hr_vis[0, t_i]
            hr_1im = util.tensor2img(hr_1im)
            save_path = osp.join(video_dir, "hr_" + str(t_i + 1) + img_ext)
            util.save_img(hr_1im, save_path)
