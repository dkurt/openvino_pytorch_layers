import torch
from packaging import version
from typing import List, Tuple, Union

def roll(
    data: torch.Tensor,
    shift: Union[int, Union[Tuple[int, ...], List[int]]],
    dims: Union[int, Union[Tuple, List]],
) -> torch.Tensor:
    """
    Similar to numpy roll but applies to pytorch tensors.
    Parameters
    ----------
    data : torch.Tensor
    shift: tuple, int
    dims : tuple, list or int

    Returns
    -------
    torch.Tensor
    """
    if isinstance(shift, (tuple, list)) and isinstance(dims, (tuple, list)):
        if len(shift) != len(dims):
            raise ValueError(f"Length of shifts and dimensions should be equal. Got {len(shift)} and {len(dims)}.")
        for curr_shift, curr_dim in zip(shift, dims):
            data = roll(data, curr_shift, curr_dim)
        return data
    dim_index = dims
    shift = shift % data.size(dims)

    if shift == 0:
        return data
    left_part = data.narrow(dim_index, 0, data.size(dims) - shift)
    right_part = data.narrow(dim_index, data.size(dims) - shift, shift)
    return torch.cat([right_part, left_part], dim=dim_index)

def fftshift(data: torch.Tensor) -> torch.Tensor:
    dim = (1, 2)
    shift = [data.size(curr_dim) // 2 for curr_dim in dim]
    return roll(data, shift, dim)

def ifftshift(data: torch.Tensor) -> torch.Tensor:
    dim = (1, 2)
    shift = [(data.size(curr_dim) + 1) // 2 for curr_dim in dim]
    return roll(data, shift, dim)

class FFT(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, inverse, centered=False):
        return g.op('IFFT' if inverse else 'FFT', x,
                    inverse_i=inverse, centered_i=centered)

    @staticmethod
    def forward(self, x, inverse, centered=False, signal_ndim=None):
        # https://pytorch.org/docs/stable/torch.html#torch.fft
        if signal_ndim is None:
            signal_ndim = 2 if len(x.shape) == 5 else 1
        if centered:
                x = ifftshift(x)

        if version.parse(torch.__version__) >= version.parse("1.8.0"):
            func = torch.fft.ifftn if inverse else torch.fft.fftn
            x = torch.view_as_complex(x)
            y = func(x, dim=list(range(1, signal_ndim + 1)), norm="ortho")
            y = torch.view_as_real(y)
        else:
            func = torch.ifft if inverse else torch.fft
            y = func(input=x, signal_ndim=signal_ndim, normalized=True)

        if centered:
            y = fftshift(y)

        return y
