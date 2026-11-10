# Interactive Document Scanner and Perspective Correction

An educational Document Scanner with image algorithms implemented from scratch in **NumPy**. OpenCV is restricted to image I/O, resizing and display annotations. No YOLO or pretrained model is used. It automatically detects document outlines in photos (even when shot at an angle), crops them, corrects perspective skew ("bird's-eye view"), and applies scanning filters. It features both a Command Line Interface (CLI) and an interactive Graphical User Interface (GUI) with draggable handles for fine-tuning.

---

## Features
- **Auto Contour Detection**: Preprocesses the image and automatically locates the document's edges using connected components, convex hulls and RDP implemented in NumPy.
- **High-Resolution Warp**: Performs perspective analysis on a resized image for speed, but applies the warp projection directly to the original high-resolution image to preserve text readability.
- **Draggable Manual Override (GUI)**: Drag green handles to adjust corners if shadows or complex backgrounds block automated boundary detection.
- **Scanned Document Filters**: Includes color, grayscale, and adaptive binary thresholding (black-and-white print look).
- **Post-Processing**: Rotates output (90° increments) and sharpens text using a high-pass filter.
- **Multi-Format Export**: Saves the final document as an image (`JPG`, `PNG`, etc.) or exports directly to a high-quality `PDF`.

---

## Installation & Setup

### 1. Prerequisites
Make sure you have Python 3.7+ installed.

### 2. Install Dependencies
Install the required packages from the project root:
```bash
pip install -r requirements.txt
```

---

## Usage

### Local contrast for faint ink / handwriting

**Clean Scanned (B&W):** the binary mode now runs grayscale -> mild Gaussian
denoising (3 x 3, sigma 0.8) -> paper background estimation (41 x 41 grayscale
closing and Gaussian smoothing) -> division by background -> optional CLAHE ->
adaptive threshold (31 x 31). Its offset is `7 * (1 + contrast)` to compensate
for amplified residual noise. All operations are implemented with NumPy.
Color and grayscale modes keep their existing appearance. Restart the GUI to
load the updated code, then choose Scanned (B&W); no new switch is required.

This targets dark ink on light paper. It reduces smooth shading and fine paper
noise but cannot restore blurred or missing strokes. Hard shadows, large dark
graphics, colored fills and extremely faint pencil marks remain difficult.
Compare with Grayscale for faint handwriting before discarding the original.
The cleanup does not delete small connected components, which could be dots or
Vietnamese diacritics. See `output/clean_scan/comparison.jpg` (original crop,
previous binary processing, clean binary processing).

In the GUI, choose **Grayscale** and start with **Contrast = 50%**.
The slider ranges from 0% (off, original behavior) to 100%. Toggle it to
compare the result. For colored ink, use **Full Color**. Try **Scanned (B&W)**
after checking that thin strokes remain visible.

```bash
python scan.py -i handwriting.jpg -o enhanced.png -m gray --contrast 0.5
python scan.py -i handwriting.jpg -o binary.png -m scanned --contrast 0.5
python -m unittest discover -s tests -v
```

Contrast uses CLAHE (clip limit 2.0, 8 x 8 tiles), blended with the original
luminance by the selected strength. For color images it computes weighted BGR luminance and adds the luminance change equally to each
channel (with saturation clipping); for grayscale it operates directly on intensity. It runs after
perspective correction and before adaptive thresholding/sharpening. Each
preview starts from the unfiltered warp, so adjusting the slider is reversible.

This improves local visibility, but does not recognize handwriting or perform
OCR. Strong contrast can also emphasize speckles; reduce the slider for noisy
scans. Binary output may lose faint strokes, so retain a grayscale copy.
Real handwriting accuracy has not been measured; synthetic thin-stroke tests
only verify contrast improvement, not OCR accuracy.

### 1. Command Line Interface (CLI)
You can scan images using `scan.py` from your terminal.

```bash
# Scan and view the step-by-step preview windows
python scan.py -i sample.jpg --show

# Save the scanned document in black-and-white scanned mode
python scan.py -i sample.jpg -o output.jpg -m scanned

# Save the warped image as color and export to PDF
python scan.py -i sample.jpg -o output.jpg --pdf output.pdf -m color --sharpen

# Rotate the scanned document by 90 degrees clockwise
python scan.py -i sample.jpg -o output.jpg -r 90
```

#### CLI Parameters:
- `-i`, `--input` (required): Path to the source photo.
- `-o`, `--output` (optional): Path to save the processed image.
- `--pdf` (optional): Path to save as a PDF document.
- `-m`, `--mode`: Output style (`color`, `gray`, or `scanned` for black & white). Default is `color`.
- `-s`, `--sharpen`: Flag to apply an unsharp mask filter to enhance text.
- `-r`, `--rotate`: Degrees to rotate clockwise (`0`, `90`, `180`, `270`).
- `--show`: Displays intermediate processing screens.

---

### 2. Graphical User Interface (GUI)
Run the GUI to interactively adjust and preview your scans:
```bash
python gui.py
```
- Click **📂 Load Image** to import any picture.
- Drag the **green circles** on the left screen to match the exact boundaries of your document.
- Use the **Sliders** on the bottom-left to fine-tune edge detection thresholds.
- Toggle between **Full Color**, **Grayscale**, and **Scanned (B&W)**.
- Adjust rotation or check **Sharpen Text**.
- Click **Save Image** or **Export PDF** to export your final document.

---

## Algorithms implemented from scratch

All computational implementations live in `image_algorithms.py`. `scanner.py`
keeps the application API and combines them. GUI and CLI use the same backend.

| Function | Implementation |
|---|---|
| `grayscale` | BGR weighted sum: 0.114 B + 0.587 G + 0.299 R |
| `gaussian_blur` | Normalized Gaussian weights, separable sliding sums |
| `canny` | Sobel gradients, magnitude/direction, non-maximum suppression, two thresholds, 8-connected hysteresis |
| `morphology`, `close_edges` | Neighborhood max/min; dilation then erosion |
| `document_contour` | Flood-fill edge components, monotone-chain convex hull, RDP simplification, shoelace area |
| `homography` | Assemble eight correspondence equations, solve using NumPy linear algebra |
| `warp_perspective` | Inverse coordinate mapping, bilinear interpolation, black outside source |
| `adaptive_threshold` | Compare intensity with Gaussian local mean minus offset |
| `clahe` | Tile histograms, clipping, excess redistribution, cumulative distributions, interpolation between tile LUTs |
| `correlate` | Sliding weighted sum used for sharpening and Sobel |
| `resize` | Optional NumPy bilinear resize; application resize may use OpenCV as permitted |

The document detector chooses the largest quadrilateral covering at least 2%
of the image. It uses the convex outer boundary, not OpenCV's full contour
hierarchy. Connected clutter can therefore merge with a page; manual corner
adjustment remains necessary for difficult images. The corner ordering uses a
cyclic convex hull to prevent duplicated corners on diamond-shaped pages.

The warp output dimensions use the largest opposing edge lengths. This is a
practical estimate, not recovery of the physical paper aspect ratio from an
uncalibrated camera. Floating-point operations, border handling, Canny magnitude
and CLAHE redistribution differ from OpenCV: outputs are not bit-identical.
Python flood-fill may be slower on highly textured images. The application
still detects on a resized preview and warps the full-resolution source.

OpenCV calls allowed in the application: image read/write, resize, display/window
handling and drawing display annotations. The sample generator uses drawing only
to create artificial source artwork; its perspective transformation is NumPy.
No OpenCV image-processing algorithm is called by the scanner.

Run `python -m unittest discover -s tests -v` (also compatible with pytest).
Tests cover known homographies, identity warps, corner permutations, synthetic
skewed-page detection, faint strokes and an AST check for forbidden cv2 calls.
The subsequent clean-scan update adds background division and mild denoising
to binary output; see Clean Scanned (B&W) above.

Historical verification of the initial NumPy migration on six `sample/*.jpg` images: all 18 CLI exports
(color/gray/scanned, contrast 0.5) completed, five images returned automatic
corners and `images.jpg` used the fallback region. This is an execution check,
not a document-detection accuracy benchmark. GUI filter callbacks passed a
hidden-window smoke test. Ten unit tests passed. See
`output/numpy_verification/comparison.jpg` for input / gray / binary columns.
Binary results on textured paper contain visible speckles; background cleanup
and tuning remain unfinished. The fallback photo still requires manual corners.

---
## Edge Cases and Pitfalls

### A. Automatic Detection Fails (No 4-Point Contour Found)
- **Cause**: Background texture matches paper color, or severe shadows break the border outline.
- **Handling**: 
  - The CLI issues a warning and defaults to a 5% margin boundary box around the center.
  - The GUI permits the user to drag the handles to the correct corners manually.
  - Users can adjust parameters using edge tuners to make margins clear.

### B. Multiple Quadrilaterals Found
- **Cause**: Other large rectangular objects (e.g., keyboards, books, tables) are in frame.
- **Handling**: Contours are pre-sorted by area. The scanner checks the largest contour first. If it is a quadrilateral, it assumes it is the document. To avoid conflicts, keep the document as the main subject in the frame.

### C. Text is Blurred / Pixels are Stretched
- **Cause**: Detection was run on the low-resolution preview image, and warping was applied directly to that preview instead of the original source.
- **Handling**: Our system detects shapes on the resized $800\text{px}$ image for speed, scales those coordinates by the scale ratio, and executes the transformation on the original high-resolution matrix. This ensures the output text is sharp and legible.

