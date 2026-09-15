import torch

from complex_blindspot_dmri.masking import generate_paired_masks, mixed_fill


def test_masks_and_fill_preserve_shapes():
    torch.manual_seed(7)
    image = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
    target, hidden = generate_paired_masks(
        image.shape,
        image.device,
        target_probability=0.2,
        pair_probability=1.0,
    )
    filled = mixed_fill(image, target, hidden)
    assert target.shape == (2, 1, 16, 16)
    assert hidden.shape == target.shape
    assert not torch.any(target & hidden)
    assert filled.shape == image.shape
    assert filled.dtype == image.dtype
    assert torch.equal(filled[~(target | hidden).expand_as(image)], image[~(target | hidden).expand_as(image)])

