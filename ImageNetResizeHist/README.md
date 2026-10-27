# ImageNet Resizing and Histogram Enhancement

Samples 2,000 ImageNet validation images, resizes them to 256 x 256, and compares histogram enhancement in RGB, YUV, HSV, and grayscale. PyTorch uses CUDA when available and otherwise falls back to CPU.

| File | Purpose |
| --- | --- |
| `GenerateImageNet2000.py` | Sample and rename validation images |
| `ImageResizer.py` | Nearest-neighbor and bilinear interpolation |
| `HistogramEqualizer.py` | Global histogram equalization |
| `HistogramSpecifier.py` | Specification using a sigmoid target CDF |
| `HistogramMatcher.py` | Matching to a reference image |
| `CLAHEProcessor.py` | Contrast-limited adaptive histogram equalization |
| `main.py` | Full pipeline and MSE, PSNR, SSIM, contrast, and entropy evaluation |

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
python main.py
```

Provide at least 2,000 `.JPEG` images in `dataset/ImageNetVal/` and a reference image at `dataset/reference.JPEG`. Sampling is random and has no fixed seed. Paths, batch size, and sample count are configured in the scripts; evaluation expects IDs `0001` through `2000`.

The pipeline writes processed images, comparison strips, `evaluation_results.csv`, and `evaluation_summary.xlsx` under `dataset/`. Input data and generated results are excluded from Git.
