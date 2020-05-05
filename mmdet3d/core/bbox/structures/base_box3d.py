from abc import abstractmethod

import torch


class BaseInstance3DBoxes(object):
    """Base class for 3D Boxes

    """

    def __init__(self, tensor, box_dim=7):
        """
        Args:
            tensor (torch.Tensor | np.ndarray): a Nxbox_dim matrix.
            box_dim (int): number of the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
        """
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(
                dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()
        self.box_dim = box_dim
        self.tensor = tensor

    @abstractmethod
    def volume(self):
        """Computes the volume of all the boxes.

        Returns:
            torch.Tensor: a vector with volume of each box.
        """
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @abstractmethod
    def bottom_center(self):
        """Calculate the bottom center of all the boxes.

        Returns:
            torch.Tensor: a tensor with center of each box.
        """
        return self.tensor[..., :3]

    @abstractmethod
    def gravity_center(self):
        """Calculate the gravity center of all the boxes.

        Returns:
            torch.Tensor: a tensor with center of each box.
        """
        pass

    @abstractmethod
    def corners(self):
        """Calculate the coordinates of corners of all the boxes.

        Returns:
            torch.Tensor: a tensor with 8 corners of each box.
        """
        pass

    @abstractmethod
    def rotate(self, angles, axis=0):
        """Calculate whether the points is in any of the boxes

        Args:
            angles (float): rotation angles
            axis (int): the axis to rotate the boxes
        """
        pass

    @abstractmethod
    def flip(self):
        """Flip the boxes in horizontal direction
        """
        pass

    @abstractmethod
    def translate(self, trans_vector):
        """Calculate whether the points is in any of the boxes

        Args:
            trans_vector (torch.Tensor): translation vector of size 1x3
        """
        pass

    @abstractmethod
    def in_range(self, box_range):
        """Check whether the boxes are in the given range

        Args:
            box_range (list | torch.Tensor): the range of box
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Returns:
            a binary vector, indicating whether each box is inside
            the reference range.
        """
        pass

    def nonempty(self, threshold: float = 0.0):
        """Find boxes that are non-empty.

        A box is considered empty,
        if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = ((size_x > threshold)
                & (size_y > threshold) & (size_z > threshold))
        return keep

    def scale(self, scale_factors):
        """Scale the box with horizontal and vertical scaling factors

        Args:
            scale_factors (float | torch.Tensor | list[float]):
                scale factors to scale the boxes.
        """
        pass

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`:
                return a `Boxes` that contains only one box.
            2. `new_boxes = boxes[2:10]`:
                return a slice of boxes.
            3. `new_boxes = boxes[vector]`:
                where vector is a torch.BoolTensor with `length = len(boxes)`.
                Nonzero elements in the vector will be selected.
            Note that the returned Boxes might share storage with this Boxes,
            subject to Pytorch's indexing semantics.

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b)

    def __len__(self):
        return self.tensor.shape[0]

    def __repr__(self):
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, boxes_list):
        """Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])
        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, cls) for box in boxes_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    def to(self, device):
        original_type = type(self)
        return original_type(self.tensor.to(device))

    def clone(self):
        """Clone the Boxes.

        Returns:
            Boxes
        """
        original_type = type(self)
        return original_type(self.tensor.clone())

    @property
    def device(self):
        return self.tensor.device

    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor
