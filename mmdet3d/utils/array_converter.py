# Copyright (c) OpenMMLab. All rights reserved.
import functools
from inspect import getfullargspec
from typing import Callable, Optional, Tuple, Type, Union

import numpy as np
import torch

TemplateArrayType = Union[np.ndarray, torch.Tensor, list, tuple, int, float]


def array_converter(to_torch: bool = True,
                    apply_to: Tuple[str, ...] = tuple(),
                    template_arg_name_: Optional[str] = None,
                    recover: bool = True) -> Callable:
    """Wrapper function for data-type agnostic processing.

    First converts input arrays to PyTorch tensors or NumPy arrays for middle
    calculation, then convert output to original data-type if `recover=True`.

    Args:
        to_torch (bool): Whether to convert to PyTorch tensors for middle
            calculation. Defaults to True.
        apply_to (Tuple[str]): The arguments to which we apply data-type
            conversion. Defaults to an empty tuple.
        template_arg_name_ (str, optional): Argument serving as the template
            (return arrays should have the same dtype and device as the
            template). Defaults to None. If None, we will use the first
            argument in `apply_to` as the template argument.
        recover (bool): Whether or not to recover the wrapped function outputs
            to the `template_arg_name_` type. Defaults to True.

    Raises:
        ValueError: When template_arg_name_ is not among all args, or when
            apply_to contains an arg which is not among all args, a ValueError
            will be raised. When the template argument or an argument to
            convert is a list or tuple, and cannot be converted to a NumPy
            array, a ValueError will be raised.
        TypeError: When the type of the template argument or an argument to
            convert does not belong to the above range, or the contents of such
            an list-or-tuple-type argument do not share the same data type, a
            TypeError will be raised.

    Returns:
        Callable: Wrapped function.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Use torch addition for a + b,
        >>> # and convert return values to the type of a
        >>> @array_converter(apply_to=('a', 'b'))
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> a = np.array([1.1])
        >>> b = np.array([2.2])
        >>> simple_add(a, b)
        >>>
        >>> # Use numpy addition for a + b,
        >>> # and convert return values to the type of b
        >>> @array_converter(to_torch=False, apply_to=('a', 'b'),
        >>>                  template_arg_name_='b')
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> simple_add(a, b)
        >>>
        >>> # Use torch funcs for floor(a) if flag=True else ceil(a),
        >>> # and return the torch tensor
        >>> @array_converter(apply_to=('a',), recover=False)
        >>> def floor_or_ceil(a, flag=True):
        >>>     return torch.floor(a) if flag else torch.ceil(a)
        >>>
        >>> floor_or_ceil(a, flag=False)
    """

    def array_converter_wrapper(func):
        """Outer wrapper for the function."""

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Inner wrapper for the arguments."""
            if len(apply_to) == 0:
                return func(*args, **kwargs)

            func_name = func.__name__

            arg_spec = getfullargspec(func)

            arg_names = arg_spec.args
            arg_num = len(arg_names)
            default_arg_values = arg_spec.defaults
            if default_arg_values is None:
                default_arg_values = []
            no_default_arg_num = len(arg_names) - len(default_arg_values)

            kwonly_arg_names = arg_spec.kwonlyargs
            kwonly_default_arg_values = arg_spec.kwonlydefaults
            if kwonly_default_arg_values is None:
                kwonly_default_arg_values = {}

            all_arg_names = arg_names + kwonly_arg_names

            # in case there are args in the form of *args
            if len(args) > arg_num:
                named_args = args[:arg_num]
                nameless_args = args[arg_num:]
            else:
                named_args = args
                nameless_args = []

            # template argument data type is used for all array-like arguments
            if template_arg_name_ is None:
                template_arg_name = apply_to[0]
            else:
                template_arg_name = template_arg_name_

            if template_arg_name not in all_arg_names:
                raise ValueError(f'{template_arg_name} is not among the '
                                 f'argument list of function {func_name}')

            # inspect apply_to
            for arg_to_apply in apply_to:
                if arg_to_apply not in all_arg_names:
                    raise ValueError(
                        f'{arg_to_apply} is not an argument of {func_name}')

            new_args = []
            new_kwargs = {}

            converter = ArrayConverter()
            target_type = torch.Tensor if to_torch else np.ndarray

            # non-keyword arguments
            for i, arg_value in enumerate(named_args):
                if arg_names[i] in apply_to:
                    new_args.append(
                        converter.convert(
                            input_array=arg_value, target_type=target_type))
                else:
                    new_args.append(arg_value)

                if arg_names[i] == template_arg_name:
                    template_arg_value = arg_value

            kwonly_default_arg_values.update(kwargs)
            kwargs = kwonly_default_arg_values

            # keyword arguments and non-keyword arguments using default value
            for i in range(len(named_args), len(all_arg_names)):
                arg_name = all_arg_names[i]
                if arg_name in kwargs:
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=kwargs[arg_name],
                            target_type=target_type)
                    else:
                        new_kwargs[arg_name] = kwargs[arg_name]
                else:
                    default_value = default_arg_values[i - no_default_arg_num]
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=default_value, target_type=target_type)
                    else:
                        new_kwargs[arg_name] = default_value
                if arg_name == template_arg_name:
                    template_arg_value = kwargs[arg_name]

            # add nameless args provided by *args (if exists)
            new_args += nameless_args

            return_values = func(*new_args, **new_kwargs)
            converter.set_template(template_arg_value)

            def recursive_recover(input_data):
                if isinstance(input_data, (tuple, list)):
                    new_data = []
                    for item in input_data:
                        new_data.append(recursive_recover(item))
                    return tuple(new_data) if isinstance(input_data,
                                                         tuple) else new_data
                elif isinstance(input_data, dict):
                    new_data = {}
                    for k, v in input_data.items():
                        new_data[k] = recursive_recover(v)
                    return new_data
                elif isinstance(input_data, (torch.Tensor, np.ndarray)):
                    return converter.recover(input_data)
                else:
                    return input_data

            if recover:
                return recursive_recover(return_values)
            else:
                return return_values

        return new_func

    return array_converter_wrapper


class ArrayConverter:
    """Utility class for data-type agnostic processing.

    Args:
        template_array (np.ndarray or torch.Tensor or list or tuple or int or
            float, optional): Template array. Defaults to None.
    """
    SUPPORTED_NON_ARRAY_TYPES = (int, float, np.int8, np.int16, np.int32,
                                 np.int64, np.uint8, np.uint16, np.uint32,
                                 np.uint64, np.float16, np.float32, np.float64)

    def __init__(self,
                 template_array: Optional[TemplateArrayType] = None) -> None:
        if template_array is not None:
            self.set_template(template_array)

    def set_template(self, array: TemplateArrayType) -> None:
        """Set template array.

        Args:
            array (np.ndarray or torch.Tensor or list or tuple or int or
                float): Template array.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to a
                NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range, or the
                contents of a list or tuple do not share the same data type, a
                TypeError is raised.
        """
        self.array_type = type(array)
        self.is_num = False
        self.device = 'cpu'

        if isinstance(array, np.ndarray):
            self.dtype = array.dtype
        elif isinstance(array, torch.Tensor):
            self.dtype = array.dtype
            self.device = array.device
        elif isinstance(array, (list, tuple)):
            try:
                array = np.array(array)
                if array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
                self.dtype = array.dtype
            except (ValueError, TypeError):
                print('The following list cannot be converted to a numpy '
                      f'array of supported dtype:\n{array}')
                raise
        elif isinstance(array, (int, float)):
            self.array_type = np.ndarray
            self.is_num = True
            self.dtype = np.dtype(type(array))
        else:
            raise TypeError(
                f'Template type {self.array_type} is not supported.')

    def convert(
        self,
        input_array: TemplateArrayType,
        target_type: Optional[Type] = None,
        target_array: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Convert input array to target data type.

        Args:
            input_array (np.ndarray or torch.Tensor or list or tuple or int or
                float): Input array.
            target_type (Type, optional): Type to which input array is
                converted. It should be `np.ndarray` or `torch.Tensor`.
                Defaults to None.
            target_array (np.ndarray or torch.Tensor, optional): Template array
                to which input array is converted. Defaults to None.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to a
                NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range, or the
                contents of a list or tuple do not share the same data type, a
                TypeError is raised.

        Returns:
            np.ndarray or torch.Tensor: The converted array.
        """
        if isinstance(input_array, (list, tuple)):
            try:
                input_array = np.array(input_array)
                if input_array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
            except (ValueError, TypeError):
                print('The input cannot be converted to a single-type numpy '
                      f'array:\n{input_array}')
                raise
        elif isinstance(input_array, self.SUPPORTED_NON_ARRAY_TYPES):
            input_array = np.array(input_array)
        array_type = type(input_array)
        assert target_type is not None or target_array is not None, \
            'must specify a target'
        if target_type is not None:
            assert target_type in (np.ndarray, torch.Tensor), \
                'invalid target type'
            if target_type == array_type:
                return input_array
            elif target_type == np.ndarray:
                # default dtype is float32
                converted_array = input_array.cpu().numpy().astype(np.float32)
            else:
                # default dtype is float32, device is 'cpu'
                converted_array = torch.tensor(
                    input_array, dtype=torch.float32)
        else:
            assert isinstance(target_array, (np.ndarray, torch.Tensor)), \
                'invalid target array type'
            if isinstance(target_array, array_type):
                return input_array
            elif isinstance(target_array, np.ndarray):
                converted_array = input_array.cpu().numpy().astype(
                    target_array.dtype)
            else:
                converted_array = target_array.new_tensor(input_array)
        return converted_array

    def recover(
        self, input_array: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor, int, float]:
        """Recover input type to original array type.

        Args:
            input_array (np.ndarray or torch.Tensor): Input array.

        Returns:
            np.ndarray or torch.Tensor or int or float: Converted array.
        """
        assert isinstance(input_array, (np.ndarray, torch.Tensor)), \
            'invalid input array type'
        if isinstance(input_array, self.array_type):
            return input_array
        elif isinstance(input_array, torch.Tensor):
            converted_array = input_array.cpu().numpy().astype(self.dtype)
        else:
            converted_array = torch.tensor(
                input_array, dtype=self.dtype, device=self.device)
        if self.is_num:
            converted_array = converted_array.item()
        return converted_array
