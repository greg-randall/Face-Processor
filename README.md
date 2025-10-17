# Face Processor

**Automatically crop, align, and pad headshots with precision using facial landmark detection.**

Face Processor standardizes headshot images by detecting facial landmarks and aligning them to a consistent position and scale. It can intelligently generate background imagery for photos that are too tightly cropped, making it perfect for creating uniform profile pictures and professional headshot galleries.

## 🚀 Windows Downloadable Release

**Ready-to-use Windows executable available!** No Python installation required.

Download the latest Windows release from the repository's Releases section, or build your own using the instructions below. The Windows executable includes all dependencies and provides the full GUI experience in a single portable application.

**Key Features:**

  - **Consistent Sizing & Alignment**: Uses the full height of the face (chin to forehead) for robust scaling, with eye-line alignment for vertical positioning.
  - **Content-Aware Fill**: Intelligently generates background for images that require padding, creating a seamless, natural look.
  - **High-Quality Resizing**: Employs the Pillow-Lanczos resampling algorithm to produce sharp, high-quality final images with minimal artifacts.
  - **Advanced Analysis**: Creates a comprehensive alignment profile from your reference images and generates visual aids (`_analysis.jpg`, `_debug.jpg`) for fine-tuning.
  - **Batch Processing**: Run on entire directories of images at once.
  - **Configurable**: Adjust aspect ratios, sizes, and alignment targets to fit any style.

-----

## Before and After

Look in the 'samples' folder for more.

| Original                                         | Cropped (Aligned & Padded)                               |
| :----------------------------------------------- | :------------------------------------------------------- |
| <img src="/samples/sample (2).jpg" width="300"> | <img src="/samples/sample (2)_cropped.jpg" width="300"> |
| <img src="/samples/sample (5).jpg" width="300"> | <img src="/samples/sample (5)_cropped.jpg" width="300"> |
| <img src="/samples/sample (1).jpg" width="300"> | <img src="/samples/sample (1)_cropped.jpg" width="300"> |

-----

## Quick Start

### Installation

#### Option 1: Windows Executable (Recommended for Windows users)
1. Download the latest `FaceProcessor_Windows.zip` from the Releases section
2. Extract the archive to your preferred location
3. Run `FaceProcessor.exe` - no installation required!

#### Option 2: Python Installation
```bash
pip install opencv-python mediapipe numpy pillow tqdm pyinstaller
```

**GUI Requirements:**
The GUI uses tkinter which comes pre-installed with most Python distributions. If not available, install it separately depending on your system.

### Basic Usage

**GUI Usage (Recommended):**

```
python gui.py
```

The GUI provides a user-friendly interface for both cropping and analysis modes with real-time progress tracking and detailed logging.

**Command Line Usage:**

**Crop a single image:**

```
python face_processor.py path/to/your/headshot.jpg
```

**Process an entire directory:**

```
python face_processor.py path/to/image/directory/
```

**Create a custom alignment profile from reference photos:**

```
python face_processor.py --analyze path/to/reference/images/
```

This will find the average position and size of faces in your existing photos, then apply that same crop style to new photos.

-----

## GUI Features

The graphical interface (`gui.py`) provides an intuitive way to process images with the following capabilities:

  - **Two Processing Modes**: Switch between "Crop Images" and "Analyze Folder" modes
  - **Real-time Progress**: Visual progress bar showing current processing status
  - **Live Logging**: Detailed output log with processing messages and status updates
  - **File/Folder Selection**: Easy browse buttons for selecting input files or directories
  - **Configurable Options**: Adjustable settings for image size, aspect ratio, and content-aware fill
  - **Toggle Log Window**: Show/hide the log output pane to conserve screen space

**GUI Controls:**

- **Input Selection**: Use "Select File..." for single images or "Select Folder..." for batch processing
- **Mode Selection**: Choose between cropping existing images or analyzing a folder to create alignment profiles
- **Cropping Options**:
  - Size (width): Set the output width in pixels (default: 600)
  - Aspect Ratio: Specify the final image ratio (e.g., "1:1", "4:5", "16:9")
  - Content-Aware Fill: Enable/disable intelligent background padding
- **Progress Tracking**: Visual indicator shows processing progress for batch operations
- **Log Output**: Detailed processing information including success messages and warnings

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
