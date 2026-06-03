"""Tests for src/physics.py."""

import torch

from src.physics import A_PHYS, D_PHYS, den_from_nphi, physics_loss


# ----------------------------------------
# Tests — den_from_nphi
# ----------------------------------------

def test_den_from_nphi_zero_nphi_gives_zero():
    """At NPHI=0, expected DEN is 0 regardless of GR (interaction term vanishes)."""
    result = den_from_nphi(torch.tensor(0.0), torch.tensor(1.0))
    assert abs(result.item()) < 1e-6


def test_den_from_nphi_zero_gr_gives_nphi_only():
    """At GR=0, interaction term vanishes: result = A * NPHI."""
    nphi = torch.tensor(1.0)
    result = den_from_nphi(nphi, torch.tensor(0.0))
    assert abs(result.item() - A_PHYS) < 1e-6


def test_den_from_nphi_interaction_term():
    """At NPHI=1, GR=1: result = A + D."""
    result = den_from_nphi(torch.tensor(1.0), torch.tensor(1.0))
    assert abs(result.item() - (A_PHYS + D_PHYS)) < 1e-6


def test_den_from_nphi_linearity_in_nphi():
    """Output must be linear in NPHI for fixed GR."""
    gr = torch.full((50,), 0.5)
    nphi = torch.linspace(0.0, 1.0, 50)
    out = den_from_nphi(nphi, gr)
    expected = A_PHYS * nphi + D_PHYS * (nphi * gr)
    torch.testing.assert_close(out, expected)


def test_den_from_nphi_custom_coefficients():
    """Custom a and d override the defaults."""
    result = den_from_nphi(torch.tensor(1.0), torch.tensor(1.0), a=2.0, d=0.5)
    assert abs(result.item() - 2.5) < 1e-6


def test_den_from_nphi_batch_shape():
    """Output shape matches input shape for any batch size."""
    for shape in [(10,), (4, 1), (8, 3)]:
        x = torch.rand(*shape)
        g = torch.rand(*shape)
        assert den_from_nphi(x, g).shape == x.shape


def test_den_from_nphi_negative_slope():
    """Default slope is negative: higher NPHI → lower expected DEN (at GR=0)."""
    low  = den_from_nphi(torch.tensor(0.2), torch.tensor(0.0))
    high = den_from_nphi(torch.tensor(0.8), torch.tensor(0.0))
    assert high < low, "expected DEN must decrease with NPHI (negative slope)"


# ----------------------------------------
# Tests — physics_loss
# ----------------------------------------

def test_physics_loss_perfect_prediction_is_zero():
    """When den_pred equals den_from_nphi(nphi, gr), loss must be zero."""
    nphi = torch.rand(20)
    gr   = torch.rand(20)
    den_perfect = den_from_nphi(nphi, gr)
    loss = physics_loss(den_perfect, nphi, gr)
    assert loss.item() < 1e-10


def test_physics_loss_nonzero_on_deviation():
    """A constant wrong prediction must produce a positive loss."""
    nphi = torch.rand(20)
    gr   = torch.rand(20)
    den_wrong = torch.full((20,), 0.5)
    loss = physics_loss(den_wrong, nphi, gr)
    assert loss.item() > 0.0


def test_physics_loss_scalar_output():
    """Return value must be a scalar tensor."""
    loss = physics_loss(torch.rand(10), torch.rand(10), torch.rand(10))
    assert loss.shape == torch.Size([])


def test_physics_loss_gradient_flows():
    """Gradients must back-propagate through den_pred."""
    nphi = torch.rand(16)
    gr   = torch.rand(16)
    den_pred = torch.rand(16, requires_grad=True)
    loss = physics_loss(den_pred, nphi, gr)
    loss.backward()
    assert den_pred.grad is not None
    assert den_pred.grad.shape == den_pred.shape


def test_physics_loss_accepts_2d_input():
    """Inputs shaped (N, 1) must work identically to (N,)."""
    nphi = torch.rand(12)
    gr   = torch.rand(12)
    den_pred = torch.rand(12)
    loss_1d = physics_loss(den_pred, nphi, gr)
    loss_2d = physics_loss(den_pred.unsqueeze(1), nphi.unsqueeze(1), gr.unsqueeze(1))
    torch.testing.assert_close(loss_1d, loss_2d)


def test_physics_loss_custom_coefficients():
    """Custom a and d are forwarded to den_from_nphi."""
    nphi = torch.rand(10)
    gr   = torch.rand(10)
    den_pred = 2.0 * nphi + 0.5 * (nphi * gr)
    loss = physics_loss(den_pred, nphi, gr, a=2.0, d=0.5)
    assert loss.item() < 1e-10


def test_physics_loss_increases_with_error():
    """Larger prediction error must yield larger loss."""
    nphi = torch.rand(30)
    gr   = torch.rand(30)
    den_expected = den_from_nphi(nphi, gr)
    loss_small = physics_loss(den_expected + 0.01, nphi, gr)
    loss_large = physics_loss(den_expected + 0.10, nphi, gr)
    assert loss_large > loss_small
