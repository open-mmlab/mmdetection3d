# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import base64
import mmcv
import numpy as np
from nuimages import NuImages
from nuimages.utils.utils import mask_decode, name_to_index_mapping
from os import path as osp

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

NAME_MAPPING = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data/nuimages',
        help='specify the root path of dataset')
    parser.add_argument(
        '--version',
        type=str,
        nargs='+',
        default=['v1.0-mini'],
        required=False,
        help='specify the dataset version')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/nuimages/annotations/',
        required=False,
        help='path to save the exported json')
    parser.add_argument(
        '--nproc',
        type=int,
        default=4,
        required=False,
        help='workers to process semantic masks')
    parser.add_argument('--extra-tag', type=str, default='nuimages')
    args = parser.parse_args()
    return args


def get_img_annos(nuim, img_info, cat2id, out_dir, data_root, seg_root):
    """Get semantic segmentation map for an image.

    Args:
        nuim (obj:`NuImages`): NuImages dataset object
        img_info (dict): Meta information of img

    Returns:
        np.ndarray: Semantic segmentation map of the image
    """
    sd_token = img_info['token']
    image_id = img_info['id']
    name_to_index = name_to_index_mapping(nuim.category)

    # Get image data.
    width, height = img_info['width'], img_info['height']
    semseg_mask = np.zeros((height, width)).astype('uint8')

    # Load stuff / surface regions.
    surface_anns = [
        o for o in nuim.surface_ann if o['sample_data_token'] == sd_token
    ]

    # Draw stuff / surface regions.
    for ann in surface_anns:
        # Get color and mask.
        category_token = ann['category_token']
        category_name = nuim.get('category', category_token)['name']
        if ann['mask'] is None:
            continue
        mask = mask_decode(ann['mask'])

        # Draw mask for semantic segmentation.
        semseg_mask[mask == 1] = name_to_index[category_name]

    # Load object instances.
    object_anns = [
        o for o in nuim.object_ann if o['sample_data_token'] == sd_token
    ]

    # Sort by token to ensure that objects always appear in the
    # instance mask in the same order.
    object_anns = sorted(object_anns, key=lambda k: k['token'])

    # Draw object instances.
    # The 0 index is reserved for background; thus, the instances
    # should start from index 1.
    annotations = []
    for i, ann in enumerate(object_anns, start=1):
        # Get color, box, mask and name.
        category_token = ann['category_token']
        category_name = nuim.get('category', category_token)['name']
        if ann['mask'] is None:
            continue
        mask = mask_decode(ann['mask'])

        # Draw masks for semantic segmentation and instance segmentation.
        semseg_mask[mask == 1] = name_to_index[category_name]

        if category_name in NAME_MAPPING:
            cat_name = NAME_MAPPING[category_name]
            cat_id = cat2id[cat_name]

            x_min, y_min, x_max, y_max = ann['bbox']
            # encode calibrated instance mask
            mask_anno = dict()
            mask_anno['counts'] = base64.b64decode(
                ann['mask']['counts']).decode()
            mask_anno['size'] = ann['mask']['size']

            data_anno = dict(
                image_id=image_id,
                category_id=cat_id,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=mask_anno,
                iscrowd=0)
            annotations.append(data_anno)

    # after process, save semantic masks
    img_filename = img_info['file_name']
    seg_filename = img_filename.replace('jpg', 'png')
    seg_filename = osp.join(seg_root, seg_filename)
    mmcv.imwrite(semseg_mask, seg_filename)
    return annotations, np.max(semseg_mask)


def export_nuim_to_coco(nuim, data_root, out_dir, extra_tag, version, nproc):
    print('Process category information')
    categories = []
    categories = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    cat2id = {k_v['name']: k_v['id'] for k_v in categories}

    images = []
    print('Process image meta information...')
    for sample_info in mmcv.track_iter_progress(nuim.sample_data):
        if sample_info['is_key_frame']:
            img_idx = len(images)
            images.append(
                dict(
                    id=img_idx,
                    token=sample_info['token'],
                    file_name=sample_info['filename'],
                    width=sample_info['width'],
                    height=sample_info['height']))

    seg_root = f'{out_dir}semantic_masks'
    mmcv.mkdir_or_exist(seg_root)
    mmcv.mkdir_or_exist(osp.join(data_root, 'calibrated'))

    global process_img_anno

    def process_img_anno(img_info):
        single_img_annos, max_cls_id = get_img_annos(nuim, img_info, cat2id,
                                                     out_dir, data_root,
                                                     seg_root)
        return single_img_annos, max_cls_id

    print('Process img annotations...')
    if nproc > 1:
        outputs = mmcv.track_parallel_progress(
            process_img_anno, images, nproc=nproc)
    else:
        outputs = []
        for img_info in mmcv.track_iter_progress(images):
            outputs.append(process_img_anno(img_info))

    # Determine the index of object annotation
    print('Process annotation information...')
    annotations = []
    max_cls_ids = []
    for single_img_annos, max_cls_id in outputs:
        max_cls_ids.append(max_cls_id)
        for img_anno in single_img_annos:
            img_anno.update(id=len(annotations))
            annotations.append(img_anno)

    max_cls_id = max(max_cls_ids)
    print(f'Max ID of class in the semantic map: {max_cls_id}')

    coco_format_json = dict(
        images=images, annotations=annotations, categories=categories)

    mmcv.mkdir_or_exist(out_dir)
    out_file = osp.join(out_dir, f'{extra_tag}_{version}.json')
    print(f'Annotation dumped to {out_file}')
    mmcv.dump(coco_format_json, out_file)


def main():
    args = parse_args()
    for version in args.version:
        nuim = NuImages(
            dataroot=args.data_root, version=version, verbose=True, lazy=True)
        export_nuim_to_coco(nuim, args.data_root, args.out_dir, args.extra_tag,
                            version, args.nproc)


if __name__ == '__main__':
    main()
