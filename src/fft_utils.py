import torch


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D FFT
    x: (..., H, W) complex
    """
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D IFFT
    k: (..., H, W) complex
    """
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    k = torch.fft.ifft2(k, norm="ortho")
    k = torch.fft.fftshift(k, dim=(-2, -1))
    return k


def center_crop(x: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    """
    x: (..., H, W)
    """
    H, W = x.shape[-2], x.shape[-1]
    top = (H - crop_h) // 2
    left = (W - crop_w) // 2
    return x[..., top:top + crop_h, left:left + crop_w]


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def normalize_instance(x: torch.Tensor, eps: float = 1e-8):
    mean = x.mean()
    std = x.std()
    x = (x - mean) / (std + eps)
    return x, mean, std


def rss_combine(coil_images: torch.Tensor, coil_dim: int = 0) -> torch.Tensor:
    """
    Root-sum-of-squares coil combination.

    Parameters
    ----------
    coil_images : torch.Tensor
        Complex coil images, commonly [coils, height, width].
    coil_dim : int
        Dimension containing the coil channels.

    Returns
    -------
    torch.Tensor
        Real-valued RSS image.
    """
    return torch.sqrt(
        torch.sum(torch.abs(coil_images) ** 2, dim=coil_dim)
        + 1e-12
    )
