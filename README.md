# Face Processor

**Turn any headshot into a perfectly aligned, professional portrait in seconds.**

## Why Face Processor?

Ever struggled with inconsistent headshots for your team page, dating profile, or portfolio? Face Processor automatically detects faces and creates perfectly aligned, consistently sized portraits - even intelligently adding background to photos that are too tightly cropped. No more manual cropping or expensive photo editing.

## See It In Action

![GUI Demo](/samples/gui.gif)

### Before and After

| Original                                         | Processed (Auto-aligned & Padded)                        |
| :----------------------------------------------- | :------------------------------------------------------- |
| <img src="/samples/sample (2).jpg" width="300"> | <img src="/samples/sample (2)_cropped.jpg" width="300"> |
| <img src="/samples/sample (5).jpg" width="300"> | <img src="/samples/sample (5)_cropped.jpg" width="300"> |
| <img src="/samples/sample (1).jpg" width="300"> | <img src="/samples/sample (1)_cropped.jpg" width="300"> |

*Look in the 'samples' folder for more examples.*

## Quick Start

### 🚀 Windows Users (No Installation!)

1. Download `FaceProcessor_Windows.zip` from [Releases](../../releases)
2. Extract and run `FaceProcessor.exe`
3. Select your images and click Process!

### 🐍 Python Users

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the GUI
python gui.py
```

That's it! Select your images in the GUI and watch the magic happen.

### Command Line Power Users

```bash
# Process a single image
python face_processor.py photo.jpg

# Process entire folder
python face_processor.py ./headshots/

# Match the style of existing photos
python face_processor.py --analyze ./reference-photos/
```

## Key Features

✨ **Smart Face Detection** - Uses MediaPipe's advanced facial landmark detection  
🎯 **Perfect Alignment** - Automatically aligns eyes and face position across all images  
🖼️ **Content-Aware Fill** - Intelligently generates background for tight crops  
📁 **Batch Processing** - Process entire folders with one click  
🎨 **Customizable Output** - Adjust aspect ratios, sizes, and alignment targets  
📊 **Style Matching** - Analyze existing photos to match their cropping style  

-----

# Full Documentation

## GUI Interface

The Face Processor GUI provides:
- **Two Modes**: "Crop Images" for processing and "Analyze Folder" for style matching
- **Real-time Progress**: Visual progress bar and live logging
- **Easy Selection**: Browse buttons for files and folders
- **Customization**: Adjust output size, aspect ratio, and content-aware fill
- **Batch Processing**: Process entire directories with one click

-----

## Command Line Arguments

### Universal Options

| Argument       | Description                    | Default        |
| :------------- | :----------------------------- | :------------- |
| `input_path`   | Path to image file or directory. | **Required** |
| `-o, --output` | Output file path or JSON file. | Auto-generated |

### Cropping Mode (Default)

| Argument            | Description                                                 | Default                   |
| :------------------ | :---------------------------------------------------------- | :------------------------ |
| `-c, --config`      | Path to a JSON file with landmark targets.                  | `median_landmarks.json`   |
| `--size`            | The width of the final cropped image in pixels.             | `600`                     |
| `--aspect-ratio`    | The aspect ratio of the final image (e.g., "1:1", "4:5").   | `"1:1"`                   |
| `--eye-y`           | Manually set the target relative Y position for the eyes.   | From config               |
| `--face-height`     | Manually set the target relative height for the face.       | From config               |
| `--no-content-fill` | Disable the content-aware fill for background padding.      | `False` (enabled by default) |

### Analysis & Debug Options

| Argument    | Description                                                                                               | Default |
| :---------- | :-------------------------------------------------------------------------------------------------------- | :------ |
| `--analyze` | Switch to analysis mode to generate a landmark profile from a directory.                                  | `False` |
| `--debug`   | Enable debug messages and save intermediate images (`_analysis.jpg`, `_debug.jpg`, `analysis_average.jpg`). | `False` |

-----

## How It Works

### Two-Step Process

**Analysis Phase (Optional)**

  - Processes a directory of reference images to calculate median landmark positions.
  - Creates a JSON profile (`median_landmarks.json`) with the optimal alignment targets.
  - Generates visual aids: an `_analysis.jpg` for each source image with markers, and a combined `analysis_average.jpg` to show the median positions.
  - Use this when you have a set of existing headshots that you want all future photos to match.

**Processing Phase**

  - Detects facial landmarks in each source image using MediaPipe's Face Mesh.
  - Calculates the ideal crop box based on the alignment targets from your config file or the script's defaults.
  - If the crop requires padding, it uses **content-aware fill** to seamlessly generate new background pixels.
  - Crops the image from the full-resolution source.
  - Resizes the final crop using a high-quality **Pillow-Lanczos** filter.
  - Outputs a consistently framed and sized headshot.

### Landmark-Based Alignment

The tool uses a landmark-driven approach for precision:

  - **Sizing**: The scale of the crop is determined by the **full vertical height of the face**—from the top of the forehead (landmark 10) to the bottom of the chin (landmark 152). This provides a very stable measurement that isn't affected by expression.
  - **Alignment**: The face is centered based on the midpoint of the pupils. This is used for both horizontal and vertical positioning, ensuring the eyes always land in the same spot.

-----

## File Handling

### Supported Formats

  - **Input**: JPG, JPEG, PNG, BMP, TIFF, WebP
  - **Output**: Same format as input
  - **Transparency**: PNG files are flattened onto a background. If possible, the background color is detected from the image's metadata.

### Output Naming

  - **Cropping**: Adds an `_cropped` suffix (e.g., `photo.jpg` -\> `photo_cropped.jpg`).
  - **Analysis**: Creates `_analysis.jpg` for each input file and a single `analysis_average.jpg`.
  - **Debug**: Creates `_debug.jpg` showing the un-resized, full-resolution crop.

-----

## Configuration Files

Custom alignment profiles are JSON files containing target positions, like this:

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

-----

## Building from Source

### Windows Development Build

For developers who want to build the Windows executable themselves:

```bash
# Clone or download the repository
git clone <repository-url>
cd Face-Processor

# Install Python dependencies
pip install -r requirements.txt

# Build the executable (Windows only)
build.bat
```

The `build.bat` script will:
1. Clean any previous build artifacts
2. Create a new PyInstaller executable using `gui.spec`
3. Package the release into a timestamped ZIP archive
4. Clean up temporary build files

### Manual Build

If you prefer manual control over the build process:

```bash
# Build the executable
python -m PyInstaller --noconfirm --clean gui.spec

# Package the release
python package_release.py
```

### Build Requirements

- **Python 3.10+** with all dependencies from `requirements.txt`
- **Windows OS** (for executable creation)
- **PyInstaller** (included in requirements.txt)

The build process creates a self-contained executable that includes:
- MediaPipe face detection models
- All Python dependencies
- The complete GUI interface
- Default configuration files

The resulting `FaceProcessor_Windows_*.zip` archive contains a portable `FaceProcessor.exe` that can be distributed to Windows users without any Python installation required.

-----

## Technical Updates

### What's New in Recent Versions

- **PyInstaller Integration**: Full support for creating standalone Windows executables
- **MediaPipe Bundle Optimization**: Essential MediaPipe assets are now included in the executable bundle
- **Improved Resource Handling**: Robust path resolution for both development and deployed environments
- **Automated Release Packaging**: Script-based creation of timestamped release archives
- **Enhanced Build Process**: Clean build workflow with proper cleanup and temp file management
