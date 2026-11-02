import torch
from kokoro.istftnet import Generator


def test_generator_compiles_without_graph_breaks():
    # Kokoro-82M's generator settings, at a small input size
    generator = Generator(
        style_dim=128, resblock_kernel_sizes=[3, 7, 11], upsample_rates=[10, 6], upsample_initial_channel=512,
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], upsample_kernel_sizes=[20, 12],
        gen_istft_n_fft=20, gen_istft_hop_size=5,
    ).eval()
    x, s, f0 = torch.randn(1, 512, 10), torch.randn(1, 128), torch.rand(1, 10) * 200
    with torch.no_grad():
        explanation = torch._dynamo.explain(generator)(x, s, f0)
    assert explanation.graph_break_count == 0
