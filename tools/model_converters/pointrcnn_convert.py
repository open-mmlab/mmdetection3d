import torch
import mmengine

mm_path = '/home/PJLAB/shenkun/openmmlab-refactor/mmdetection3d/checkpoints/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth'
pc_path = '/home/PJLAB/shenkun/workspace/OpenPCDet/checkpoint/pointrcnn_7870.pth'

def main():
    new_dict = dict()
    ori = torch.load(mm_path)
    mm_dict = torch.load(mm_path)['state_dict']
    pc_dict = torch.load(pc_path)['model_state']
    pc_dict.pop('global_step')
    for i in range(len(mm_dict.keys())):
        mm_name = list(mm_dict.keys())[i]
        if 'backbone' in mm_name and 'conv' in mm_name and 'bias' in mm_name:
            continue
        else:
            new_dict[mm_name] = mm_dict[mm_name]
    for i in range(len(new_dict.keys())):
        new_dict[list(new_dict.keys())[i]] = pc_dict[list(pc_dict.keys())[i]]
    ori['state_dict'] = new_dict
    torch.save(ori,'new_pointrcnn.pth')

if __name__ == '__main__':
    main()
