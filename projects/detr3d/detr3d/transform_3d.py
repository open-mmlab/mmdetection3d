from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class filename2img_path:

    def __call__(self, results):
        results['img_path'] = results['filename']
        return results

    def __repr__(self):
        return 'maybe we need to fix this bug'
