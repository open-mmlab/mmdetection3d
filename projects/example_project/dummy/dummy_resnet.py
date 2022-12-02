from mmdet3d.models.builder import BACKBONES
from mmdet.models.backbones import ResNet


@BACKBONES.register_module()
class DummyResNet(ResNet):
    """Implements a dummy ResNet wrapper for demonstration purpose.
    Args:
        **kwargs: All the arguments are passed to the parent class.
    """

    def __init__(self, **kwargs) -> None:
        print('Hello world!')
        super().__init__(**kwargs)
