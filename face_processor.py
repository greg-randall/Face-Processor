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

@contextmanager
def suppress_stderr():
    """
    A context manager to forcefully suppress stderr by redirecting its file descriptor.
    This is effective at silencing C-level library logs.
    """
    original_tf_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

def find_median_landmarks(image_paths, face_mesh, debug=False, output_dir=''):
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
    for file_path in tqdm(image_paths, desc=desc, unit="image", disable=debug, file=sys.stdout):
        image = cv2.imread(file_path)
        if image is None:
            if debug:
                tqdm.write(f"Warning: Could not read image at {file_path}. Skipping.")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # --- Save annotated image for this specific file ---
            base, ext = os.path.splitext(os.path.basename(file_path))
            analysis_output_path = os.path.join(output_dir, f"{base}_analysis{ext}")
            draw_analysis_markers(image.copy(), face_landmarks.landmark, analysis_output_path)
            analysis_image_paths.append(analysis_output_path)

            # --- Collect data for median calculation ---
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
                tqdm.write(f"Warning: No face detected in {file_path}. Skipping.")

    if not eye_centers or not face_heights:
        return None, []

    # --- Calculate Medians ---
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
    inpainted_this_run = False
    # --- 1. Load Image and Handle Transparency ---
    background_color = (255, 255, 255)
    if image_path.lower().endswith('.png'):
        try:
            with Image.open(image_path) as pil_img:
                if 'background' in pil_img.info:
                    rgb_bg = pil_img.info['background']
                    background_color = (rgb_bg[2], rgb_bg[1], rgb_bg[0]) # BGR for OpenCV
        except Exception:
            pass
        image_rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image_rgba is None:
            tqdm.write(f"Error: Could not read image at {image_path}")
            return inpainted_this_run
        if image_rgba.shape[2] == 4:
            alpha = image_rgba[:, :, 3]
            # Use the first pixel color as background if no other is available.
            # This helps with transparent backgrounds that aren't pure white/black.
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
        tqdm.write(f"Error: Could not read image at {image_path}")
        return inpainted_this_run

    # --- 2. Find Landmarks ---
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
        tqdm.write(f"Error: No face detected in {os.path.basename(image_path)}. Cannot crop.")
        return inpainted_this_run

    landmarks = results.multi_face_landmarks[0].landmark
    eye_center_px_y = ((landmarks[473].y + landmarks[468].y) / 2) * image_height
    eye_center_px_x = ((landmarks[473].x + landmarks[468].x) / 2) * image_width
    source_face_height_px = (landmarks[152].y - landmarks[10].y) * image_height

    # --- 3. Calculate Ideal Crop Box ---
    try:
        w, h = map(int, aspect_ratio.split(':'))
        ratio_h_w = h / w
    except ValueError:
        tqdm.write(f"Warning: Invalid aspect ratio '{aspect_ratio}'. Using 1:1.")
        ratio_h_w = 1.0

    if source_face_height_px <= 0:
        tqdm.write(f"Error: Could not calculate a valid face size in {os.path.basename(image_path)}. Skipping.")
        return inpainted_this_run

    crop_height = int(source_face_height_px / targets['face_height'])
    crop_width = int(crop_height / ratio_h_w)
    crop_x = int(eye_center_px_x - (crop_width * targets['x_center']))
    crop_y = int(eye_center_px_y - (crop_height * targets['eye_y']))
    
    # --- 4. Create Final Image (with or without inpainting) ---
    padding_needed = (crop_x < 0 or crop_y < 0 or 
                      (crop_x + crop_width) > image_width or 
                      (crop_y + crop_height) > image_height)

    if use_content_fill and padding_needed:
        # --- NEW METHOD: Inpaint a border, then crop ---
        pad_top = max(0, -crop_y)
        pad_bottom = max(0, (crop_y + crop_height) - image_height)
        pad_left = max(0, -crop_x)
        pad_right = max(0, (crop_x + crop_width) - image_width)
        
        # Add extra padding to give the inpaint algorithm more context
        extra_pad = 10 
        bordered_image = cv2.copyMakeBorder(
            image, 
            pad_top + extra_pad, pad_bottom + extra_pad, 
            pad_left + extra_pad, pad_right + extra_pad, 
            cv2.BORDER_CONSTANT, value=(0,0,0) # Pad with black for easy masking
        )
        
        mask = np.zeros(bordered_image.shape[:2], dtype=np.uint8)
        mask[pad_top + extra_pad : pad_top + extra_pad + image_height, 
             pad_left + extra_pad : pad_left + extra_pad + image_width] = 255
        mask = cv2.bitwise_not(mask)
        
        inpainted_bordered = cv2.inpaint(bordered_image, mask, 5, cv2.INPAINT_TELEA)
        inpainted_this_run = True

        # Calculate crop coordinates within the new, bordered image
        final_crop_x = crop_x + pad_left + extra_pad
        final_crop_y = crop_y + pad_top + extra_pad
        
        initial_crop = inpainted_bordered[final_crop_y : final_crop_y + crop_height,
                                          final_crop_x : final_crop_x + crop_width]
    else:
        # --- OLD METHOD: Paste onto a solid color canvas ---
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

    # --- 4a. Save debug image if requested ---
    if debug:
        base, ext = os.path.splitext(output_path)
        debug_output_path = base.replace('_cropped', '_debug') + ext
        if initial_crop.shape[0] > 0 and initial_crop.shape[1] > 0:
            cv2.imwrite(debug_output_path, initial_crop)

    # --- 5. Resize with Pillow and Save ---
    if initial_crop.shape[0] <= 0 or initial_crop.shape[1] <= 0:
        tqdm.write(f"Error: Created an empty crop for {os.path.basename(image_path)}. Skipping save.")
        return False
        
    target_width = size
    target_height = int(target_width * ratio_h_w)

    # Convert OpenCV image (BGR) to Pillow image (RGB)
    rgb_crop = cv2.cvtColor(initial_crop, cv2.COLOR_BGR2RGB)
    pillow_image = Image.fromarray(rgb_crop)

    # Resize using Pillow's Lanczos filter
    resized_pillow_image = pillow_image.resize((target_width, target_height), Image.LANCZOS)
    
    # Convert back to OpenCV format to save
    resized_rgb_array = np.array(resized_pillow_image)
    final_image = cv2.cvtColor(resized_rgb_array, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, final_image)
    if not quiet_success:
        tqdm.write(f"✅ Successfully saved: {os.path.basename(output_path)}")

    return inpainted_this_run

# --- Action Handlers ---
def run_analysis(args):
    """Runs the logic to analyze images and generate a profile."""
    image_directory = args.input_path
    output_filename = args.output if args.output else 'median_landmarks.json'

    if not os.path.isdir(image_directory):
        print(f"❌ Error: Input for analysis must be a directory. Path not found: '{image_directory}'")
        return

    image_files = []
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    for ext in supported_extensions:
        image_files.extend(glob.glob(os.path.join(image_directory, ext)))

    # Filter out our own analysis and debug files from being processed
    image_files = [
        f for f in image_files 
        if not f.lower().endswith(f'_analysis{os.path.splitext(f)[1]}') 
        and not f.lower().endswith(f'_debug{os.path.splitext(f)[1]}')
    ]


    if not image_files:
        print(f"❌ Error: No supported images found in '{image_directory}'")
        return
    
    print(f"--- Analyzing {len(image_files)} images to create alignment profile ---")

    median_landmarks = None
    analysis_image_paths = []
    
    def process_images_with_mesh():
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        # Pass the output directory to the analysis function
        median_data, generated_files = find_median_landmarks(image_files, face_mesh, debug=args.debug, output_dir=image_directory)
        face_mesh.close()
        return median_data, generated_files

    if not args.debug:
        with suppress_stderr():
            median_landmarks, analysis_image_paths = process_images_with_mesh()
    else:
        median_landmarks, analysis_image_paths = process_images_with_mesh()

    if median_landmarks:
        print("\n--- Median Landmark Positions ---")
        print("These values are relative to the image dimensions (0.0 to 1.0).")
        print(f"Median Eye Center:     (x={median_landmarks['median_eye_center']['x']:.4f}, y={median_landmarks['median_eye_center']['y']:.4f})")
        print(f"Median Forehead Top:   (x={median_landmarks['median_forehead_top']['x']:.4f}, y={median_landmarks['median_forehead_top']['y']:.4f})")
        print(f"Median Chin Bottom:    (x={median_landmarks['median_chin_bottom']['x']:.4f}, y={median_landmarks['median_chin_bottom']['y']:.4f})")
        print(f"Median Relative Face Height: {median_landmarks['median_face_height']:.4f}")

        try:
            with open(output_filename, 'w') as json_file:
                json.dump(median_landmarks, json_file, indent=4)
            print(f"\n✅ Successfully saved landmark data to {output_filename}")
        except IOError as e:
            print(f"\n❌ Error: Could not write to file {output_filename}. Reason: {e}")

        # --- Create and save the average image ---
        if analysis_image_paths:
            print("\n--- Creating average analysis image ---")
            target_size = (600, 600)
            accumulator = np.zeros((target_size[1], target_size[0], 3), np.float32)

            for path in tqdm(analysis_image_paths, desc="Averaging Images", unit="image"):
                img = cv2.imread(path)
                if img is not None:
                    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    accumulator += resized_img
            
            if len(analysis_image_paths) > 0:
                average_image = (accumulator / len(analysis_image_paths)).astype(np.uint8)

                # Draw median lines on the average image
                h, w, _ = average_image.shape
                forehead_y = int(median_landmarks['median_forehead_top']['y'] * h)
                chin_y = int(median_landmarks['median_chin_bottom']['y'] * h)

                cv2.line(average_image, (0, forehead_y), (w, forehead_y), (0, 0, 255), 2) # Red line for forehead
                cv2.line(average_image, (0, chin_y), (w, chin_y), (0, 0, 255), 2) # Red line for chin

                avg_output_path = os.path.join(image_directory, 'analysis_average.jpg')
                cv2.imwrite(avg_output_path, average_image)
                print(f"✅ Successfully saved average analysis to {avg_output_path}")

    elif image_files:
        print("\nNo faces were detected in any of the processed images.")

def run_cropping(args):
    """Runs the logic for the default cropping action."""
    did_inpaint_occur = False # Flag to track if any inpainting happened

    # --- Configuration Loading ---
    targets = {
        'eye_y': 0.45,
        'face_height': 0.65, 
        'x_center': 0.5
    }
    print("--- Setting Alignment Targets ---")
    try:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                data = json.load(f)
                targets['eye_y'] = data['median_eye_center']['y']
                targets['face_height'] = data['median_face_height']
                targets['x_center'] = data['median_eye_center']['x']
            print(f"✅ Loaded targets from '{args.config}'")
        else:
            if args.config != "median_landmarks.json":
                print(f"⚠️  Warning: Specified config file not found: '{args.config}'")
            print("ℹ️  Using built-in default targets.")
    except Exception as e:
        print(f"⚠️  Warning: Could not read or parse '{args.config}'. Using defaults. Error: {e}")

    # Check for command-line overrides
    if args.eye_y is not None:
        targets['eye_y'] = args.eye_y
        print(f"✅ Overriding eye-y with command-line value: {targets['eye_y']}")
    if args.face_height is not None:
        targets['face_height'] = args.face_height
        print(f"✅ Overriding face-height with command-line value: {targets['face_height']}")

    # --- Collect Image Paths ---
    image_paths_to_process = []
    if os.path.isfile(args.input_path):
        image_paths_to_process.append(args.input_path)
    elif os.path.isdir(args.input_path):
        if args.output is not None:
            print("⚠️  Warning: --output flag is ignored when processing a directory.")
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        all_files_in_dir = glob.glob(os.path.join(args.input_path, '*'))
        for file_path in all_files_in_dir:
            if os.path.isfile(file_path) and file_path.lower().endswith(supported_extensions):
                base, _ = os.path.splitext(os.path.basename(file_path))
                # Updated to ignore _debug files
                if not base.lower().endswith('_cropped') and not base.lower().endswith('_analysis') and not base.lower().endswith('_debug'):
                    image_paths_to_process.append(file_path)
    else:
        print(f"❌ Error: Input path is not a valid file or directory: {args.input_path}")
        return

    # --- Run Cropping Loop ---
    if not image_paths_to_process:
        print("\n🤷 No new images found to process.")
        return

    total_images = len(image_paths_to_process)
    if total_images > 1:
        print(f"\nFound {total_images} images to process.")

    use_content_fill = args.content_fill

    if os.path.isdir(args.input_path):
        for image_path in tqdm(image_paths_to_process, desc="Cropping Images", unit="image"):
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_cropped{ext}"
            inpainted = crop_and_align_headshot(
                image_path, output_path, args.size, args.aspect_ratio, 
                targets, use_content_fill, quiet_success=True, debug=args.debug
            )
            if inpainted:
                did_inpaint_occur = True
    else:
        for image_path in image_paths_to_process:
            print(f"\n--- Cropping: {os.path.basename(image_path)} ---")
            output_path = args.output
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_cropped{ext}"
            inpainted = crop_and_align_headshot(
                image_path, output_path, args.size, args.aspect_ratio, 
                targets, use_content_fill, quiet_success=False, debug=args.debug
            )
            if inpainted:
                did_inpaint_occur = True

    # --- Final Warning ---
    if did_inpaint_occur:
        print("\n" + "="*40)
        print("XXXHAD TO CONTENT AWARE FILLXXXX")
        print("="*40)
        print("\n⚠️  Warning: Some images were padded using content-aware fill.")
        print("   Please review the cropped images carefully for any artifacts.")

def main():
    """Parses arguments and runs the appropriate action."""
    parser = argparse.ArgumentParser(
        description="A tool to automatically crop and align headshots.",
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("input_path", help="Path to an image/directory to crop.")

    # --- Mode Flag ---
    parser.add_argument(
        '--analyze',
        action='store_true',
        help="Switch to analysis mode. Analyzes all images in the input directory\nto generate a landmark profile instead of cropping."
    )
    
    # --- Universal Arguments ---
    parser.add_argument(
        "-o", "--output",
        help="CROPPING: The output file path (for a single input image).\nANALYSIS: The output path for the JSON profile.",
        default=None
    )
    
    # --- Cropping-Specific Arguments ---
    crop_group = parser.add_argument_group('Cropping Options')
    crop_group.add_argument("-c", "--config", default="median_landmarks.json", help='Path to a JSON file with landmark targets for cropping.')
    crop_group.add_argument("--size", type=int, default=600, help="The width of the final cropped image in pixels. (Default: 600)")
    crop_group.add_argument("--aspect-ratio", type=str, default="1:1", help='The aspect ratio of the final image, e.g., "1:1", "4:5". (Default: "1:1")')
    crop_group.add_argument("--eye-y", type=float, default=None, help="Manually set the target relative Y position for the eyes.")
    crop_group.add_argument("--face-height", type=float, default=None, help="Manually set the target relative height for the face.")
    crop_group.add_argument(
        '--no-content-fill',
        dest='content_fill',
        action='store_false',
        help="Disable the content-aware fill for background padding."
    )
    
    # --- Analysis-Specific Arguments ---
    # The debug flag now also applies to cropping mode
    analysis_group = parser.add_argument_group('Analysis & Debug Options')
    analysis_group.add_argument('--debug', action='store_true', help='Enable debug/warning messages and save intermediate debug images.')

    args = parser.parse_args()

    if hasattr(args, 'mouth_y'):
        pass
    
    if args.analyze:
        run_analysis(args)
    else:
        if 'mouth_y' in args:
           delattr(args, 'mouth_y')
        run_cropping(args)

if __name__ == '__main__':
    main()