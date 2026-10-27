# HW4: Logic, Spatial Filtering, and Edges

Demonstrates binary AND, OR, XOR, and NOT with a circular mask; mean and Gaussian smoothing; and Sobel, Prewitt, and Laplacian responses. Results appear in a 3-by-4 comparison grid.

- `image_operations.py`: masks, logical operations, spatial filters, and edge operators.
- `main.py`: image loading and the combined demo.

## Run

Install the root requirements and provide `TestData/test_plane.jpg`, or edit its path in `main.py`. Run from this directory:

```bash
python main.py
```

The spatial filters use Python loops and may be slow on large images. The function named `laplacian_sharpening` returns a clipped Laplacian response.
