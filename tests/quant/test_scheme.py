import pytest

from holograph.quant import Scheme


@pytest.mark.parametrize(
    ("bits", "group_size", "num_children", "field"),
    [(0, 64, 4, "bits"), (9, 64, 4, "bits"), (4, 0, 4, "group_size"), (4, 64, 0, "num_children")],
)
def test_rejects_invalid_settings(bits, group_size, num_children, field):
    with pytest.raises(ValueError, match=field):
        Scheme(bits=bits, group_size=group_size, num_children=num_children)
