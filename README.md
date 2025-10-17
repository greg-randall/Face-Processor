# Face Processor

**Easily align and standardize your headshots in seconds.**

## Why Face Processor?

If you’ve ever struggled with a batch of headshots that need to look consistent Face Processor can help. It automatically detects faces and aligns them to a consistent size, positioning, and aspect ratio. It even handles those photos that are too tightly cropped, filling in the background seamlessly.

## See It In Action

![GUI Demo](/samples/gui.gif)

### Before and After

| Original                                        | Processed (Auto-aligned & Padded)                       |
| :---------------------------------------------- | :------------------------------------------------------ |
| <img src="/samples/sample (2).jpg" width="300"> | <img src="/samples/sample (2)_cropped.jpg" width="300"> |
| <img src="/samples/sample (5).jpg" width="300"> | <img src="/samples/sample (5)_cropped.jpg" width="300"> |
| <img src="/samples/sample (1).jpg" width="300"> | <img src="/samples/sample (1)_cropped.jpg" width="300"> |

*Check out the 'samples' folder for more examples.*

## Quick Start

### 🚀 Windows Users (No Installation Required)

1. Download `FaceProcessor_Windows.zip` from [Releases](../../releases)
2. Extract and run `FaceProcessor.exe`
3. Select your images and click "Process"

### 🐍 Python Users

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the GUI
python gui.py
```

Then, just select your images in the GUI and let it do the work!

### Command Line Power Users

```bash
# Process a single image
python face_processor.py photo.jpg

# Process an entire folder
python face_processor.py ./headshots/

# Match the style of existing photos
python face_processor.py --analyze ./reference-photos/
```

## Key Features

✨ **Smart Face Detection** – Detects and aligns faces using MediaPipe
🎯 **Perfect Alignment** – Aligns eyes and face positions consistently across all images
🖼️ **Content-Aware Fill** – Adds background to tight crops
📁 **Batch Processing** – Process multiple images at once
🎨 **Customizable Output** – Adjust aspect ratio, size, and alignment
📊 **Style Matching** – Analyze a set of images to match their cropping style

---

# Full Documentation

## GUI Interface

The Face Processor GUI includes:

* **Two Modes**: "Crop Images" (for processing) and "Analyze Folder" (for style matching)
* **Real-time Progress**: Visual progress bar and logs
* **Simple Selection**: Browse buttons for selecting files or folders
* **Customization**: Adjust output size, aspect ratio, and content-aware fill
* **Batch Processing**: Process entire directories in one go

---

## Command Line Arguments

### Universal Options

| Argument       | Description                      | Default        |
| :------------- | :------------------------------- | :------------- |
| `input_path`   | Path to image file or directory. | **Required**   |
| `-o, --output` | Output file path or JSON file.   | Auto-generated |

### Cropping Mode (Default)

| Argument            | Description                                            | Default                      |
| :------------------ | :----------------------------------------------------- | :--------------------------- |
| `-c, --config`      | Path to a JSON file with landmark targets.             | `median_landmarks.json`      |
| `--size`            | Width of the final cropped image.                      | `600`                        |
| `--aspect-ratio`    | Aspect ratio for the final image (e.g., "1:1", "4:5"). | `"1:1"`                      |
| `--eye-y`           | Manually set the target Y position for the eyes.       | From config                  |
| `--face-height`     | Manually set the target height for the face.           | From config                  |
| `--no-content-fill` | Disable content-aware fill for background padding.     | `False` (enabled by default) |

### Analysis & Debug Options

| Argument    | Description                                                                                                 | Default |
| :---------- | :---------------------------------------------------------------------------------------------------------- | :------ |
| `--analyze` | Switch to analysis mode to generate a landmark profile from a directory.                                    | `False` |
| `--debug`   | Enable debug messages and save intermediate images (`_analysis.jpg`, `_debug.jpg`, `analysis_average.jpg`). | `False` |

---

## How It Works

### Two-Step Process

**Analysis Phase (Optional)**

* If you have a set of headshots with a specific style, run this phase to create a JSON profile of ideal alignment targets.
* This creates a profile (`median_landmarks.json`) to guide the alignment of other images.
* Visual markers and a combined `analysis_average.jpg` show the optimal alignment.

**Processing Phase**

* Detects facial landmarks using MediaPipe’s Face Mesh.
* Automatically adjusts cropping based on alignment targets, ensuring faces are consistently sized and positioned.
* Uses **content-aware fill** to seamlessly add background when needed.
* Outputs a final image with consistent framing, even for images that were initially misaligned.

### Landmark-Based Alignment

Face Processor uses a landmark-driven approach for precision:

* **Sizing**: The face’s vertical height (from forehead to chin) determines the crop size.
* **Alignment**: Faces are aligned based on the midpoint of the pupils for consistent positioning.

---

## File Handling

### Supported Formats

* **Input**: JPG, JPEG, PNG, BMP, TIFF, WebP
* **Output**: Same format as input
* **Transparency**: PNG files are flattened onto a background, with the background color inferred from image metadata.

### Output Naming

* **Cropped**: Adds `_cropped` to the original file name.
* **Analysis**: Creates `_analysis.jpg` for each input file, and `analysis_average.jpg`.
* **Debug**: Generates `_debug.jpg` showing un-resized, full-resolution crops.

---

## Configuration Files

Custom alignment profiles are stored as JSON files, like this:

```json
{
    "median_eye_center": {
        "x": 0.5014,
        "y": 0.4086
    },
    "median_face_height": 0.4361,
    "median_forehead_top": {
        "x": 0.4962,
        "y": 0.2753
    },
    "median_chin_bottom": {
        "x": 0.5056,
        "y": 0.7121
    }
}
```

---

## Building from Source

### Windows Development Build

To build the Windows executable from source:

```bash
# Clone or download the repo
git clone <repository-url>
cd Face-Processor

# Install dependencies
pip install -r requirements.txt

# Build the executable (Windows only)
build.bat
```

The `build.bat` script will create a self-contained executable that includes all necessary dependencies, including the face detection models.