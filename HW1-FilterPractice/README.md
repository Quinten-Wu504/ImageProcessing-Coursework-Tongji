# HW1: Frequency-Domain Filtering

Compares ideal, Gaussian, and Butterworth low-pass and high-pass filters using FFTs. The main demo adds Gaussian noise and requests 60% and 90% spectral-energy targets.

- `filter_utils.py`: filter masks and energy-based parameter selection.
- `main.py`: noise generation and visual comparisons.
- `test.py`: separate fixed-cutoff Gaussian filter demo; its percentage labels are illustrative.

## Run

Install the root requirements, then run from this directory:

```bash
python main.py
```

Provide `TestData/test_high_freq.jpg` (any readable image), or edit `img_path` in `main.py`. The optional `test.py` requires `TestData/cat_test.jpg`. Close each plot window to continue. Gaussian and Butterworth filters use fallback parameters when an energy target cannot be bracketed.
