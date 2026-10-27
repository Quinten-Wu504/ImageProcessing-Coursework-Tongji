# HW2: Histogram Equalization

Implements CDF-based histogram equalization and compares three modes: the Y channel in YUV, the V channel in HSV, and grayscale. Plots show the images and histograms before and after processing.

- `hist_utils.py`: equalization and visualization.
- `img_equalized.py`: color-space conversion and channel processing.
- `main.py`: runs all three modes.

## Run

Install the root requirements and provide `TestData/test_colorful.jpg`, or change the input path in `main.py`. Run from this directory:

```bash
python main.py
```

Close the plot windows to advance to the next mode.
