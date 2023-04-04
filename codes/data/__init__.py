'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    '''
    Create a dataloader given a dataset.
    '''
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
            
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        batch_size = dataset_opt['batch_size']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False,
                                           pin_memory=False)


def create_dataset(dataset_opt):
    '''
    Create a dataset given a dataset option.
    '''
    mode = dataset_opt['mode']
    print("f")
    print(mode)
    if mode == 'LQ':
        from data.LQ_dataset import LQDataset as D
    elif mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
    elif mode == 'LQGTVID':
        from data.LQGTVID_dataset import LQGTVIDDataset as D
    elif mode == 'LQGTVID_Conseutive':
        from data.LQGTVID_Conseutive_dataset import LQGTVIDDataset as D
    elif mode == 'LQGTVID_Aug':
        from data.LQGTVID_Aug_dataset import LQGTVID_AugDataset as D
    
    elif mode == 'LQGTVID_bicubic':
        from data.LQGTVID_bicubic_dataset import LQGTVIDDataset as D
    elif mode == 'LQGTVID_SR':
        from data.LQGTVID_SR_dataset import LQGTVIDDataset as D
    elif mode == 'UVG':
        from data.UVG_dataset import UVGDataset as D
    # elif mode == 'LQGTseg_bg':
    #     from data.LQGT_seg_bg_dataset import LQGTSeg_BG_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)
    print("f")
    logger = logging.getLogger('base')
    print("f")

    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
