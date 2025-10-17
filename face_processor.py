import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import json
import glob
import sys
from contextlib import contextmanager
from PIL import Image
from tqdm import tqdm

# --- Utility Functions ---

def get_image_paths(path):
    """Gets a list of image paths from a file or directory."""
    image_paths_to_process = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    if os.path.isfile(path):
        if path.lower().endswith(supported_extensions):
            image_paths_to_process.append(path)
    elif os.path.isdir(path):
        all_files = glob.glob(os.path.join(path, '*'))
        for f in all_files:
            # Filter out already processed files or analysis markers
            if f.lower().endswith(supported_extensions) and not f.lower().endswith(
                ('_cropped.jpg', '_analysis.jpg', '_debug.jpg', 
                 '_cropped.png', '_analysis.png', '_debug.png',
                 '_cropped.webp', '_analysis.webp', '_debug.webp')):
                image_paths_to_process.append(f)
    return image_paths_to_process

@contextmanager
def suppress_stderr():
    """
    A context manager to forcefully suppress stderr by redirecting its file descriptor.
    This is effective at silencing C-level library logs.
    """
    original_tf_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Check if stderr has a fileno, which it won't in the GUI's redirected output
    if not hasattr(sys.stderr, 'fileno'):
        yield
        return
    original_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull_fd, original_stderr_fd)
    try:
        yield
    finally:
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(devnull_fd)
        os.close(saved_stderr_fd)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_tf_log_level

# --- Drawing Helper ---

def draw_analysis_markers(image, landmarks, output_path):
    """Draws markers on an image for analysis and saves it."""
    h, w, _ = image.shape
    # Key landmarks
    FOREHEAD_TOP_LANDMARK = 10
    CHIN_BOTTOM_LANDMARK = 152
    LEFT_EYE_PUPIL = 473
    RIGHT_EYE_PUPIL = 468

    # Draw forehead and chin markers (larger, in red)
    forehead_pt = (int(landmarks[FOREHEAD_TOP_LANDMARK].x * w), int(landmarks[FOREHEAD_TOP_LANDMARK].y * h))
    chin_pt = (int(landmarks[CHIN_BOTTOM_LANDMARK].x * w), int(landmarks[CHIN_BOTTOM_LANDMARK].y * h))
    cv2.circle(image, forehead_pt, radius=10, color=(0, 0, 255), thickness=-1)
    cv2.circle(image, chin_pt, radius=10, color=(0, 0, 255), thickness=-1)

    # Draw eye markers (smaller, in green)
    left_eye_pt = (int(landmarks[LEFT_EYE_PUPIL].x * w), int(landmarks[LEFT_EYE_PUPIL].y * h))
    right_eye_pt = (int(landmarks[RIGHT_EYE_PUPIL].x * w), int(landmarks[RIGHT_EYE_PUPIL].y * h))
    cv2.circle(image, left_eye_pt, radius=8, color=(0, 255, 0), thickness=-1)
    cv2.circle(image, right_eye_pt, radius=8, color=(0, 255, 0), thickness=-1)

    cv2.imwrite(output_path, image)

# --- Core Logic for Analysis ---

def find_median_landmarks(image_paths, face_mesh, debug=False, output_dir='', disable_progress_bar=False, progress_callback=None):
    """
    Analyzes images to find median landmarks, saves annotated images,
    and returns data for averaging.
    """
    eye_centers = []
    face_heights = []
    forehead_tops = []
    chin_bottoms = []
    analysis_image_paths = []

    # Key landmarks
    FOREHEAD_TOP_LANDMARK = 10
    CHIN_BOTTOM_LANDMARK = 152

    desc = "Analyzing Images & Creating Markers"
    for file_path in tqdm(image_paths, desc=desc, unit="image", disable=disable_progress_bar):
        image = cv2.imread(file_path)
        if image is None:
            if debug:
                print(f"Warning: Could not read image at {file_path}. Skipping.")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            base, ext = os.path.splitext(os.path.basename(file_path))
            analysis_output_path = os.path.join(output_dir, f"{base}_analysis{ext}")
            draw_analysis_markers(image.copy(), face_landmarks.landmark, analysis_output_path)
            analysis_image_paths.append(analysis_output_path)

            left_eye_pupil = face_landmarks.landmark[473]
            right_eye_pupil = face_landmarks.landmark[468]
            forehead_top = face_landmarks.landmark[FOREHEAD_TOP_LANDMARK]
            chin_bottom = face_landmarks.landmark[CHIN_BOTTOM_LANDMARK]

            eye_centers.append([(left_eye_pupil.x + right_eye_pupil.x) / 2.0, (left_eye_pupil.y + right_eye_pupil.y) / 2.0])
            face_heights.append(chin_bottom.y - forehead_top.y)
            forehead_tops.append([forehead_top.x, forehead_top.y])
            chin_bottoms.append([chin_bottom.x, chin_bottom.y])
        else:
            if debug:
                print(f"Warning: No face detected in {file_path}. Skipping.")

        if progress_callback:
            progress_callback()

    if not eye_centers or not face_heights:
        return None, []

    median_data = {
        "median_eye_center": {"x": np.median(np.array(eye_centers), axis=0)[0], "y": np.median(np.array(eye_centers), axis=0)[1]},
        "median_face_height": np.median(np.array(face_heights)),
        "median_forehead_top": {"x": np.median(np.array(forehead_tops), axis=0)[0], "y": np.median(np.array(forehead_tops), axis=0)[1]},
        "median_chin_bottom": {"x": np.median(np.array(chin_bottoms), axis=0)[0], "y": np.median(np.array(chin_bottoms), axis=0)[1]}
    }
    
    return median_data, analysis_image_paths


# --- Core Logic for Cropping ---
def crop_and_align_headshot(image_path, output_path, size, aspect_ratio, targets, use_content_fill=True, quiet_success=False, debug=False):
    """
    Crops and aligns a headshot based on facial landmarks to a target composition.
    Returns True if content-aware fill was used, False otherwise.
    """
    print(f"Processing: {os.path.basename(image_path)}")
    inpainted_this_run = False
    background_color = (255, 255, 255)
    if image_path.lower().endswith('.png'):
        try:
            with Image.open(image_path) as pil_img:
                if 'background' in pil_img.info:
                    rgb_bg = pil_img.info['background']
                    background_color = (rgb_bg[2], rgb_bg[1], rgb_bg[0])
        except Exception:
            pass
        image_rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image_rgba is None:
            print(f"  - Error: Could not read image file.")
            return inpainted_this_run
        if image_rgba.shape[2] == 4:
            alpha = image_rgba[:, :, 3]
            auto_background_color = image_rgba[0, 0, :3]
            background = np.full(image_rgba.shape[:2], 1, dtype=np.uint8)[:, :, np.newaxis] * auto_background_color[np.newaxis, np.newaxis, :]
            image_bgr = image_rgba[:, :, :3]
            alpha_mask = cv2.merge([alpha, alpha, alpha]) / 255.0
            image = (image_bgr * alpha_mask + background * (1 - alpha_mask)).astype(np.uint8)
        else:
            image = image_rgba
    else:
        image = cv2.imread(image_path)
    if image is None:
        print(f"  - Error: Could not read image file.")
        return inpainted_this_run

    mp_face_mesh = mp.solutions.face_mesh
    image_height, image_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = None
    face_mesh = None
    try:
        with suppress_stderr():
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
            results = face_mesh.process(rgb_image)
    finally:
        if face_mesh:
            face_mesh.close()

    if not results or not results.multi_face_landmarks:
        print(f"  - Error: No face detected. Cannot crop.")
        return inpainted_this_run
    
    print("  - Face detected, calculating crop.")
    landmarks = results.multi_face_landmarks[0].landmark
    eye_center_px_y = ((landmarks[473].y + landmarks[468].y) / 2) * image_height
    eye_center_px_x = ((landmarks[473].x + landmarks[468].x) / 2) * image_width
    source_face_height_px = (landmarks[152].y - landmarks[10].y) * image_height

    try:
        w, h = map(int, aspect_ratio.split(':'))
        ratio_h_w = h / w
    except ValueError:
        print(f"  - Warning: Invalid aspect ratio '{aspect_ratio}'. Using 1:1.")
        ratio_h_w = 1.0

    if source_face_height_px <= 0:
        print(f"  - Error: Could not calculate a valid face size. Skipping.")
        return inpainted_this_run

    crop_height = int(source_face_height_px / targets['face_height'])
    crop_width = int(crop_height / ratio_h_w)
    crop_x = int(eye_center_px_x - (crop_width * targets['x_center']))
    crop_y = int(eye_center_px_y - (crop_height * targets['eye_y']))
    
    padding_needed = (crop_x < 0 or crop_y < 0 or 
                      (crop_x + crop_width) > image_width or 
                      (crop_y + crop_height) > image_height)

    if use_content_fill and padding_needed:
        print("  - Applying content-aware fill for padding.")
        pad_top = max(0, -crop_y)
        pad_bottom = max(0, (crop_y + crop_height) - image_height)
        pad_left = max(0, -crop_x)
        pad_right = max(0, (crop_x + crop_width) - image_width)
        extra_pad = 10 
        bordered_image = cv2.copyMakeBorder(
            image, 
            pad_top + extra_pad, pad_bottom + extra_pad, 
            pad_left + extra_pad, pad_right + extra_pad, 
            cv2.BORDER_CONSTANT, value=(0,0,0)
        )
        mask = np.zeros(bordered_image.shape[:2], dtype=np.uint8)
        mask[pad_top + extra_pad : pad_top + extra_pad + image_height, 
             pad_left + extra_pad : pad_left + extra_pad + image_width] = 255
        mask = cv2.bitwise_not(mask)
        inpainted_bordered = cv2.inpaint(bordered_image, mask, 5, cv2.INPAINT_TELEA)
        inpainted_this_run = True
        final_crop_x = crop_x + pad_left + extra_pad
        final_crop_y = crop_y + pad_top + extra_pad
        initial_crop = inpainted_bordered[final_crop_y : final_crop_y + crop_height,
                                          final_crop_x : final_crop_x + crop_width]
    else:
        initial_crop = np.full((crop_height, crop_width, 3), background_color, dtype=np.uint8)
        src_x_start = max(0, crop_x)
        src_y_start = max(0, crop_y)
        src_x_end = min(image_width, crop_x + crop_width)
        src_y_end = min(image_height, crop_y + crop_height)
        dst_x_start = max(0, -crop_x)
        dst_y_start = max(0, -crop_y)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        image_roi = image[src_y_start:src_y_end, src_x_start:src_x_end]
        if image_roi.size > 0:
            initial_crop[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image_roi

    if debug:
        base, ext = os.path.splitext(output_path)
        debug_output_path = base.replace('_cropped', '_debug') + ext
        if initial_crop.shape[0] > 0 and initial_crop.shape[1] > 0:
            cv2.imwrite(debug_output_path, initial_crop)

    if initial_crop.shape[0] <= 0 or initial_crop.shape[1] <= 0:
        print(f"  - Error: Created an empty crop. Skipping save.")
        return False
        
    target_width = size
    target_height = int(target_width * ratio_h_w)
    rgb_crop = cv2.cvtColor(initial_crop, cv2.COLOR_BGR2RGB)
    pillow_image = Image.fromarray(rgb_crop)
    resized_pillow_image = pillow_image.resize((target_width, target_height), Image.LANCZOS)
    resized_rgb_array = np.array(resized_pillow_image)
    final_image = cv2.cvtColor(resized_rgb_array, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, final_image)
    if not quiet_success:
        print(f"✅ Successfully saved: {os.path.basename(output_path)}\n")

    return inpainted_this_run

# --- Action Handlers ---
def run_analysis(options):
    """Runs the logic to analyze images and generate a profile."""
    image_directory = options['input_path']
    output_filename = options.get('output') if options.get('output') else 'median_landmarks.json'

    if not os.path.isdir(image_directory):
        print(f"❌ Error: Input for analysis must be a directory. Path not found: '{image_directory}'")
        return

    image_files = get_image_paths(image_directory)

    if not image_files:
        print(f"❌ Error: No supported images found in '{image_directory}'")
        return
    
    print(f"--- Analyzing {len(image_files)} images to create alignment profile ---")

    median_landmarks = None
    analysis_image_paths = []
    
    def process_images_with_mesh():
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        
        median_data, generated_files = find_median_landmarks(
            image_files, 
            face_mesh, 
            debug=options.get('debug', False), 
            output_dir=image_directory,
            disable_progress_bar=options.get('disable_progress_bar', False),
            progress_callback=options.get('progress_callback')
        )
        face_mesh.close()
        return median_data, generated_files

    if not options.get('debug', False):
        with suppress_stderr():
            median_landmarks, analysis_image_paths = process_images_with_mesh()
    else:
        median_landmarks, analysis_image_paths = process_images_with_mesh()

    if median_landmarks:
        print("\n--- Median Landmark Positions ---")
        print(f"Median Eye Center:     (x={median_landmarks['median_eye_center']['x']:.4f}, y={median_landmarks['median_eye_center']['y']:.4f})")
        print(f"Median Face Height: {median_landmarks['median_face_height']:.4f}")

        try:
            with open(output_filename, 'w') as json_file:
                json.dump(median_landmarks, json_file, indent=4)
            print(f"\n✅ Successfully saved landmark data to {output_filename}")
        except IOError as e:
            print(f"\n❌ Error: Could not write to file {output_filename}. Reason: {e}")

        if analysis_image_paths:
            print("\n--- Creating average analysis image ---")
            target_size = (600, 600)
            accumulator = np.zeros((target_size[1], target_size[0], 3), np.float32)

            disable_bar = options.get('disable_progress_bar', False)
            for path in tqdm(analysis_image_paths, desc="Averaging Images", unit="image", disable=disable_bar):
                img = cv2.imread(path)
                if img is not None:
                    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    accumulator += resized_img
            
            if len(analysis_image_paths) > 0:
                average_image = (accumulator / len(analysis_image_paths)).astype(np.uint8)
                avg_output_path = os.path.join(image_directory, 'analysis_average.jpg')
                cv2.imwrite(avg_output_path, average_image)
                print(f"✅ Successfully saved average analysis to {avg_output_path}")

    elif image_files:
        print("\nNo faces were detected in any of the processed images.")

def run_cropping(options):
    """Runs the logic for the default cropping action."""
    did_inpaint_occur = False
    progress_callback = options.get('progress_callback')

    targets = {'eye_y': 0.4, 'face_height': 0.45, 'x_center': 0.5}
    print("--- Setting Alignment Targets ---")
    try:
        config_path = options.get('config', 'median_landmarks.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = json.load(f)
                targets['eye_y'] = data['median_eye_center']['y']
                targets['face_height'] = data['median_face_height']
                targets['x_center'] = data['median_eye_center']['x']
            print(f"✅ Loaded targets from '{config_path}'")
        else:
            print(f"⚠️  Warning: Config file '{config_path}' not found. Using defaults.")
    except Exception as e:
        print(f"⚠️  Warning: Could not parse '{options.get('config')}'. Using defaults. Error: {e}")

    if options.get('eye_y') is not None:
        targets['eye_y'] = options['eye_y']
    if options.get('face_height') is not None:
        targets['face_height'] = options['face_height']

    image_paths_to_process = get_image_paths(options.get('input_path', '.'))

    if not image_paths_to_process:
        print("\n🤷 No new images found to process.")
        return

    total_images = len(image_paths_to_process)
    if total_images > 1:
        print(f"\nFound {total_images} images to process.\n")

    use_content_fill = options.get('content_fill', True)
    disable_bar = options.get('disable_progress_bar', False)
    
    # --- Process a Directory ---
    if total_images > 1:
        # For the GUI, we don't use the tqdm progress bar itself, but we iterate.
        # The detailed printing is now inside crop_and_align_headshot.
        loop_iterator = tqdm(image_paths_to_process, desc="Cropping Images", unit="image", disable=disable_bar)
        for image_path in loop_iterator:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_cropped{ext}"
            inpainted = crop_and_align_headshot(
                image_path, output_path, options.get('size', 600), options.get('aspect_ratio', '1:1'), 
                targets, use_content_fill, 
                quiet_success=False,  # Always print success for the GUI log
                debug=options.get('debug', False)
            )
            if inpainted:
                did_inpaint_occur = True
            if progress_callback:
                progress_callback()
    # --- Process a Single File ---
    elif total_images == 1:
        image_path = image_paths_to_process[0]
        output_path = options.get('output')
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_cropped{ext}"
        inpainted = crop_and_align_headshot(
            image_path, output_path, options.get('size', 600), options.get('aspect_ratio', '1:1'), 
            targets, use_content_fill, quiet_success=False, debug=options.get('debug', False)
        )
        if inpainted:
            did_inpaint_occur = True
        if progress_callback:
            progress_callback()

    if did_inpaint_occur:
        print("\n⚠️  Warning: Some images were padded using content-aware fill.")

def main():
    """Parses arguments and runs the appropriate action."""
    parser = argparse.ArgumentParser(
        description="A tool to automatically crop and align headshots.",
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("input_path", help="Path to an image/directory to process.")
    parser.add_argument('--analyze', action='store_true', help="Switch to analysis mode.")
    parser.add_argument("-o", "--output", help="Output file path.", default=None)
    
    crop_group = parser.add_argument_group('Cropping Options')
    crop_group.add_argument("-c", "--config", default="median_landmarks.json", help='Path to landmark target JSON file.')
    crop_group.add_argument("--size", type=int, default=600, help="Width of the final image in pixels.")
    crop_group.add_argument("--aspect-ratio", type=str, default="1:1", help='Aspect ratio, e.g., "1:1", "4:5".')
    crop_group.add_argument("--eye-y", type=float, help="Manual override for eye Y position.")
    crop_group.add_argument("--face-height", type=float, help="Manual override for face height.")
    crop_group.add_argument('--no-content-fill', dest='content_fill', action='store_false', help="Disable content-aware fill.")
    
    analysis_group = parser.add_argument_group('Analysis & Debug Options')
    analysis_group.add_argument('--debug', action='store_true', help='Enable debug messages and save intermediate images.')

    args = parser.parse_args()
    options = vars(args)

    if options.get('analyze'):
        run_analysis(options)
    else:
        run_cropping(options)


if __name__ == '__main__':
    main()

