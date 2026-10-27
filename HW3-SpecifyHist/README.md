# HW3: Histogram Specification

Maps a grayscale source image toward a reference image histogram using CDF lookup tables. Displays the source, reference, and result with their histograms.

- `hist_utils.py`: CDF calculation, intensity mapping, and visualization.
- `main.py`: loads both images and runs the comparison.

## Run

Install the root requirements. Provide `TestData/test_image.jpg` and `TestData/target_image.jpg`, or edit their paths in `main.py`. Run from this directory:

```bash
python main.py
```
