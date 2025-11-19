import tensorflow as tf
import json
import os

# -----------------------------------------------------------------------------
# Background Preloading
# -----------------------------------------------------------------------------
def load_background_metadata(background_folders, default_bg_width_cm=30.0, default_placement_area_pct=(0.1, 0.1, 0.9, 0.9)):
    """Loads background metadata from image_data.json files.
    
    If no image_data.json exists or if an image is missing from the metadata,
    default values are used for bg_width_cm and placement_area.
    
    Args:
        background_folders: List of folder paths containing background images
        default_bg_width_cm: Default width in cm to use when metadata is missing
        default_placement_area_pct: Default placement area as percentage of image dimensions (x_min%, y_min%, x_max%, y_max%)
    
    Returns:
        List of background metadata dictionaries
    """
    background_metadata = []
    
    for folder in background_folders:
        # First try to load existing metadata from JSON
        json_path = os.path.join(folder, "image_data.json")
        existing_metadata = {}
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                bg_data = json.load(f)
                for bg in bg_data:
                    bg["folder"] = folder  # Store folder path
                    existing_metadata[bg["image_name"]] = bg
                background_metadata.extend(bg_data)
        
        # Now find all PNG and JPG files in the folder
        image_files = [f for f in os.listdir(folder) if any(f.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])]
        for image_file in image_files:
            # Skip if we already have metadata for this image
            if image_file in existing_metadata:
                continue
                
            # Need to create default metadata for this image
            img_path = os.path.join(folder, image_file)
            try:
                # Use TensorFlow to get image dimensions
                image = tf.io.read_file(img_path)
                if image_file.lower().endswith(('.jpg', '.jpeg')):
                    image = tf.image.decode_jpeg(image, channels=3)
                else:
                    image = tf.image.decode_png(image, channels=3)
                height = tf.shape(image)[0].numpy()
                width = tf.shape(image)[1].numpy()
                
                # Calculate default placement area based on percentages
                x_min = int(width * default_placement_area_pct[0])
                y_min = int(height * default_placement_area_pct[1])
                x_max = int(width * default_placement_area_pct[2])
                y_max = int(height * default_placement_area_pct[3])
                
                # Create metadata entry
                bg_metadata = {
                    "image_name": image_file,
                    "folder": folder,
                    "bg_width_cm": default_bg_width_cm,
                    "placement_area": [x_min, y_min, x_max, y_max]
                }
                background_metadata.append(bg_metadata)
            except Exception as e:
                print(f"Error loading image {image_file}: {e}")

    # Sort by image_name to ensure video frames are in correct order
    background_metadata.sort(key=lambda x: x["image_name"])
    
    return background_metadata

def preload_backgrounds(metadata):
    """
    Preloads background images into memory along with their widths and placement areas.
    This avoids repeated disk I/O during composite generation.
    """
    images = []
    widths = []
    placement_areas = []
    for bg in metadata:
        folder = bg["folder"]
        path = os.path.join(folder, bg["image_name"])
        image = tf.io.read_file(path)
        if bg["image_name"].lower().endswith(('.jpg', '.jpeg')):
            image = tf.image.decode_jpeg(image, channels=3)
        else:
            image = tf.image.decode_png(image, channels=3)
        images.append(image)
        widths.append(bg["bg_width_cm"])
        placement_areas.append(bg["placement_area"])  # Expected format: [x_min, y_min, x_max, y_max]
    return images, widths, placement_areas

def select_random_background(overlay=False, background_images=None, bg_widths=None, bg_placement_areas=None, is_video=True):
    """
    Selects a random background image from the preloaded list.
    Uses tf.py_function to bridge between Python and the TensorFlow graph.
    If overlay, return a second background with near index
    """
    idx = tf.random.uniform([], 0, len(background_images), dtype=tf.int32)

    def _select(idx_, bg_imgs=background_images, bg_w=bg_widths, bg_areas=bg_placement_areas):
        idx_ = int(idx_)
        return bg_imgs[idx_], bg_w[idx_], bg_areas[idx_]
    
    bg_image, bg_width, placement_area = tf.py_function(
        _select, [idx], [tf.uint8, tf.float32, tf.int32]
    )

    if overlay:
        if is_video:
            # Video: use adjacent frame
            overlay_idx = tf.random.uniform([], -1, 1, dtype=tf.int32) + idx
            overlay_idx = tf.clip_by_value(overlay_idx, 0, len(background_images) - 1)
        else:
            # Static: use same frame
            overlay_idx = idx
        overlay_bg_image, _, _ = tf.py_function(
            _select, [overlay_idx], [tf.uint8, tf.float32, tf.int32]
        )
        overlay_bg_image.set_shape([None, None, 3])
    else:
        overlay_bg_image = None
        
    bg_image.set_shape([None, None, 3])
    bg_width.set_shape([])          # Scalar
    placement_area.set_shape([4])   # [x_min, y_min, x_max, y_max]
    return bg_image, bg_width, placement_area, overlay_bg_image


# -----------------------------------------------------------------------------
# Object Placement Functions
# -----------------------------------------------------------------------------
def update_mask_matrix(mask, obj_image, x, y):
    """Updates the mask with the object's pixels placed at (x, y), handling out-of-bounds placements."""
    bg_h = tf.shape(mask)[0]
    bg_w = tf.shape(mask)[1]
    obj_h = tf.shape(obj_image)[0]
    obj_w = tf.shape(obj_image)[1]
    
    # Calculate valid regions for the object and background
    # For x coordinate
    obj_x_start = tf.maximum(0, -x)
    obj_x_end = tf.minimum(obj_w, bg_w - x)
    bg_x_start = tf.maximum(0, x)
    bg_x_end = tf.minimum(bg_w, x + obj_w)
    
    # For y coordinate
    obj_y_start = tf.maximum(0, -y)
    obj_y_end = tf.minimum(obj_h, bg_h - y)
    bg_y_start = tf.maximum(0, y)
    bg_y_end = tf.minimum(bg_h, y + obj_h)
    
    # Check if any part of the object is visible
    is_visible = tf.logical_and(
        tf.less(obj_x_start, obj_x_end),
        tf.less(obj_y_start, obj_y_end)
    )
    
    def update_visible_part():
        # Extract visible part of the object
        obj_visible = obj_image[obj_y_start:obj_y_end, obj_x_start:obj_x_end, :]
        
        # Use alpha channel to determine object presence
        alpha_threshold = 50  # Values from 0-255, adjust as needed
        obj_mask = tf.cast(obj_visible[..., 3] > alpha_threshold, tf.int32)
        
        # Create a background-sized mask with zeros
        new_mask = tf.identity(mask)
        
        # Update the mask where the object is visible
        indices = tf.stack([
            tf.reshape(tf.range(bg_y_start, bg_y_end), [-1, 1]) + tf.zeros([bg_y_end - bg_y_start, bg_x_end - bg_x_start], tf.int32),
            tf.reshape(tf.range(bg_x_start, bg_x_end), [1, -1]) + tf.zeros([bg_y_end - bg_y_start, bg_x_end - bg_x_start], tf.int32)
        ], axis=-1)
        
        updates = tf.where(obj_mask > 0, tf.ones_like(obj_mask), tf.zeros_like(obj_mask))
        
        # Use scatter_nd to update the mask
        obj_mask_reshaped = tf.reshape(updates, [-1])
        indices_reshaped = tf.reshape(indices, [-1, 2])
        mask_updates = tf.scatter_nd(indices_reshaped, obj_mask_reshaped, tf.shape(mask))
        new_mask = tf.maximum(new_mask, mask_updates)
        
        return new_mask
    
    # Only update if any part is visible
    return tf.cond(is_visible, update_visible_part, lambda: mask)

def check_overlap_tf(mask, obj_image, x, y, max_overlap_pct):
    """Checks if placing the object at (x, y) would exceed allowed overlap."""
    bg_h = tf.shape(mask)[0]
    bg_w = tf.shape(mask)[1]
    
    # Use alpha channel to determine object presence
    alpha_threshold = 50  # Values from 0-255, adjust as needed
    obj_mask = tf.cast(obj_image[..., 3] > alpha_threshold, tf.int32)
    obj_mask = tf.expand_dims(obj_mask, axis=-1)
    
    # Get object dimensions
    obj_h = tf.shape(obj_mask)[0]
    obj_w = tf.shape(obj_mask)[1]
    
    # Validate coordinates - object must fit within background
    valid_coords = tf.logical_and(
        tf.logical_and(x >= 0, y >= 0),
        tf.logical_and(x + obj_w <= bg_w, y + obj_h <= bg_h)
    )
    
    # Only proceed if coordinates are valid
    def do_check():
        obj_padded = tf.image.pad_to_bounding_box(obj_mask, y, x, bg_h, bg_w)
        obj_padded = tf.squeeze(obj_padded, axis=-1)
        overlap_pixels = tf.reduce_sum(tf.cast((mask + obj_padded) > 1, tf.float32))
        total_obj_pixels = tf.reduce_sum(tf.cast(obj_padded > 0, tf.float32))
        overlap_percentage = tf.cond(
            total_obj_pixels > 0,
            lambda: overlap_pixels / total_obj_pixels * 100.0,
            lambda: 0.0
        )
        return overlap_percentage <= max_overlap_pct
    
    # Return result based on validity of coordinates
    return tf.cond(valid_coords, do_check, lambda: tf.constant(False))

def place_object(background, obj_image, x, y):
    """Blends the object image onto the background at (x, y) using alpha channel.
    
    Args:
        background: RGB background image tensor
        obj_image: RGBA object image tensor
        x, y: Coordinates for placement
    
    Returns:
        Blended image with proper alpha transparency
    """
    bg_h = tf.shape(background)[0]
    bg_w = tf.shape(background)[1]
    
    # Extract RGB and alpha channels
    obj_rgb = obj_image[..., :3]
    obj_alpha = tf.cast(obj_image[..., 3:4], tf.float32) / 255.0
    obj_alpha = tf.where(obj_alpha > 0.3, 1, 0)
    
    # Pad object and alpha mask to background size
    obj_padded = tf.image.pad_to_bounding_box(obj_rgb, y, x, bg_h, bg_w)
    alpha_padded = tf.image.pad_to_bounding_box(obj_alpha, y, x, bg_h, bg_w)
    
    # Alpha blending: result = foreground * alpha + background * (1 - alpha)
    obj_float = tf.cast(obj_padded, tf.float32)
    bg_float = tf.cast(background, tf.float32)
    blended = alpha_padded * obj_float + (1 - alpha_padded) * bg_float
    
    return tf.cast(blended, tf.uint8)

def calculate_blending_params(blending_strength):
    """
    Calculate blending parameters from a single blending_strength value (0-1).
    
    Args:
        blending_strength: Float 0-1, where:
            - 0.0 = sharp, crisp objects (minimal blending)
            - 0.5 = balanced blending
            - 1.0 = maximum blur and transparency
    
    Returns:
        dict with blur_radius, blur_iterations, alpha_threshold, min_alpha
    """
    # Blur radius: 1-9 pixels
    blur_radius = int(1 + blending_strength * 8)
    
    # Blur iterations: 0-3
    blur_iterations = int(blending_strength * 3)
    
    # Alpha threshold: 0.1-0.5 (lower = more pixels get enhanced)
    alpha_threshold = 0.5 - (blending_strength * 0.4)
    
    # Min alpha: 0.9-0.3 (higher = more opaque)
    min_alpha = 0.9 - (blending_strength * 0.6)
    
    return {
        'blur_radius': blur_radius,
        'blur_iterations': blur_iterations,
        'alpha_threshold': alpha_threshold,
        'min_alpha': min_alpha
    }

def place_object_with_blending(background, obj_image, x, y, blur_radius=5, blur_iterations=2, alpha_threshold=0.3, min_alpha=0.5):
    """Blends an RGBA object onto a background, handling out-of-bounds placements."""
    bg_h = tf.shape(background)[0]
    bg_w = tf.shape(background)[1]
    obj_h = tf.shape(obj_image)[0]
    obj_w = tf.shape(obj_image)[1]
    
    # Calculate valid regions for the object and background
    obj_x_start = tf.maximum(0, -x)
    obj_x_end = tf.minimum(obj_w, bg_w - x)
    bg_x_start = tf.maximum(0, x)
    bg_x_end = tf.minimum(bg_w, x + obj_w)
    
    obj_y_start = tf.maximum(0, -y)
    obj_y_end = tf.minimum(obj_h, bg_h - y)
    bg_y_start = tf.maximum(0, y)
    bg_y_end = tf.minimum(bg_h, y + obj_h)
    
    # Check if any part of the object is visible
    is_visible = tf.logical_and(
        tf.less(obj_x_start, obj_x_end),
        tf.less(obj_y_start, obj_y_end)
    )
    
    def blend_visible_part():
        # Extract visible part of the object
        obj_visible = obj_image[obj_y_start:obj_y_end, obj_x_start:obj_x_end, :]
        
        # Extract RGB and Alpha channels
        obj_rgb = obj_visible[..., :3]
        obj_alpha = tf.cast(obj_visible[..., 3:4], tf.float32) / 255.0
        
        # Use configurable thresholds
        mask_visible = obj_alpha > alpha_threshold
        obj_alpha = tf.where(
            mask_visible,
            tf.maximum(obj_alpha, min_alpha),  # Enhance visible pixels
            obj_alpha  # Leave transparent pixels
        )
        
        # Create a copy of the original alpha for later use
        orig_alpha = tf.identity(obj_alpha)
        
        # Add dimensions for pooling
        alpha_blur = tf.expand_dims(tf.squeeze(obj_alpha, -1), 0)
        alpha_blur = tf.expand_dims(alpha_blur, -1)
        
        # Apply multiple blur iterations for a smoother gradient
        kernel_size = [1, blur_radius, blur_radius, 1]
        for _ in range(blur_iterations):
            alpha_blur = tf.nn.avg_pool2d(
                alpha_blur,
                ksize=kernel_size,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
        
        # Remove batch dimension, keep channel dimension
        alpha_blur = tf.squeeze(alpha_blur, 0)
        
        # Create final alpha: original alpha + blurred edges only
        final_alpha = tf.minimum(orig_alpha, alpha_blur)
        
        # Extract the region of background to blend with
        bg_region = background[bg_y_start:bg_y_end, bg_x_start:bg_x_end, :]
        
        # Alpha blending: result = foreground * alpha + background * (1 - alpha)
        obj_float = tf.cast(obj_rgb, tf.float32)
        bg_float = tf.cast(bg_region, tf.float32)
        blended = final_alpha * obj_float + (1 - final_alpha) * bg_float
        
        # Create a copy of the background
        result = tf.identity(background)
        
        # Update the background with the blended region
        updates = tf.cast(blended, tf.uint8)
        result = tf.tensor_scatter_nd_update(
            result,
            tf.reshape(tf.stack([
                tf.range(bg_y_start, bg_y_end)[:, tf.newaxis] * tf.ones([bg_y_end - bg_y_start, bg_x_end - bg_x_start], tf.int32),
                tf.range(bg_x_start, bg_x_end)[tf.newaxis, :] * tf.ones([bg_y_end - bg_y_start, bg_x_end - bg_x_start], tf.int32)
            ], axis=-1), [-1, 2]),
            tf.reshape(updates, [-1, 3])
        )
        
        return result
    
    # Only blend if any part is visible
    return tf.cond(is_visible, blend_visible_part, lambda: background)

def _rotate_image_helper(image, angle, interpolation='BILINEAR'):
    """Rotates an image around its center using an affine transformation."""
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    center_y, center_x = height / 2, width / 2
    cos_angle = tf.cos(angle)
    sin_angle = tf.sin(angle)
    transform = tf.stack([
        cos_angle, -sin_angle,
        center_x - center_x * cos_angle + center_y * sin_angle,
        sin_angle, cos_angle,
        center_y - center_x * sin_angle - center_y * cos_angle,
        0.0, 0.0
    ])
    transform = tf.reshape(transform, [8])
    image = tf.expand_dims(image, 0)
    output = tf.raw_ops.ImageProjectiveTransformV2(
        images=image,
        transforms=tf.expand_dims(transform, 0),
        output_shape=[tf.cast(height, tf.int32), tf.cast(width, tf.int32)],
        interpolation=interpolation.upper()
    )
    return tf.squeeze(output, 0)

def resize_and_rotate(image, min_size, max_size, bg_ppcm, params):
    """
    Resizes and rotates an object image without cutting off corners.
    
    Args:
        image (tf.Tensor): Object image.
        min_size (float): Minimum size in cm.
        max_size (float): Maximum size in cm.
        bg_ppcm (float): Pixels per cm for the background.
    
    Returns:
        tf.Tensor: Resized and rotated object image.
    """
    obj_h = tf.cast(tf.shape(image)[0], tf.float32)
    obj_w = tf.cast(tf.shape(image)[1], tf.float32)
    
    # Calculate target size based on cm parameters
    target_diag_cm = tf.random.uniform([], min_size, max_size)
    target_diag_px = target_diag_cm * bg_ppcm
    original_diag = tf.sqrt(obj_h**2 + obj_w**2)
    scale_factor = target_diag_px / original_diag
    
    # Calculate new dimensions
    new_h = tf.maximum(obj_h * scale_factor, 1.0)
    new_w = tf.maximum(obj_w * scale_factor, 1.0)
    
    # Resize the image
    img_resized = tf.image.resize(image, [tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)])
    
    # Calculate padding needed to avoid corner clipping during rotation
    # For a square image, the worst-case diagonal is sqrt(2) times the side length
    # So we need sqrt(2)-1 times the side length as padding on each side
    padding_factor = 0.5  # This gives enough space for rotation without clipping
    
    # Calculate padding dimensions
    pad_h = tf.cast(new_h * padding_factor, tf.int32)
    pad_w = tf.cast(new_w * padding_factor, tf.int32)
    
    # Pad the image symmetrically
    paddings = [[pad_h, pad_h], [pad_w, pad_w], [0, 0]]
    padded_image = tf.pad(img_resized, paddings, mode='constant', constant_values=0)
    
    # Apply rotation to the padded image
    angle = tf.random.uniform([], *params['rotation_range'])
    rotated_image = _rotate_image_helper(padded_image, angle, interpolation='bilinear')
    
    # Calculate bounding box to crop the object to its actual size
    # Find non-zero alpha channel pixels (for RGBA images)
    if tf.shape(rotated_image)[-1] == 4:
        mask = rotated_image[..., 3] > 0
    else:
        # For RGB images, use any non-black pixel
        mask = tf.reduce_any(rotated_image > 0, axis=-1)
    
    # Use dynamic shape for mask
    indices = tf.where(mask)
    
    # Handle case when no pixels are found
    def crop_image():
        # Get min/max coordinates to define bounding box
        min_y = tf.reduce_min(indices[:, 0])
        max_y = tf.reduce_max(indices[:, 0])
        min_x = tf.reduce_min(indices[:, 1])
        max_x = tf.reduce_max(indices[:, 1])
        
        # Calculate new height and width
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        # Crop to bounding box
        cropped = rotated_image[min_y:min_y+height, min_x:min_x+width, :]
        return cropped
    
    def return_original():
        return rotated_image
    
    # Only crop if we have valid pixels
    final_image = tf.cond(
        tf.shape(indices)[0] > 0,
        crop_image,
        return_original
    )
    
    return final_image

# -----------------------------------------------------------------------------
# Object Datasets
# -----------------------------------------------------------------------------
def load_and_preprocess_image(file_path):
    """
    Loads images with alpha channel support for transparent backgrounds.
    
    Args:
        file_path: Path to the image file
        target_size: Optional resize dimensions
        
    Returns:
        A tensor with shape [height, width, 4] for RGBA images
    """
    image = tf.io.read_file(file_path)
    
    # Convert file path to lowercase string
    file_path_str = tf.strings.lower(file_path)
    
    # Use regex to check file extensions
    is_jpeg = tf.logical_or(
        tf.strings.regex_full_match(file_path_str, ".*\\.jpg$"),
        tf.strings.regex_full_match(file_path_str, ".*\\.jpeg$")
    )
    
    def decode_jpeg():
        # JPEG doesn't support transparency
        return tf.image.decode_jpeg(image, channels=3)
    
    def decode_png():
        # PNG can have transparency
        return tf.image.decode_png(image, channels=4)
    
    # Decode based on file type
    decoded_image = tf.cond(is_jpeg, decode_jpeg, decode_png)
    
    # For PNGs with 4 channels (RGBA), keep the alpha channel
    # For JPEGs with 3 channels (RGB), add an alpha channel of all 255 (fully opaque)
    has_alpha = tf.equal(tf.shape(decoded_image)[-1], 4)
    
    def add_alpha():
        # Add alpha channel of all 255 (fully opaque) to RGB images
        return tf.concat([
            decoded_image,
            tf.ones([tf.shape(decoded_image)[0], tf.shape(decoded_image)[1], 1], 
                   dtype=decoded_image.dtype) * 255
        ], axis=-1)
    
    # Ensure all images have alpha channel
    rgba_image = tf.cond(has_alpha, lambda: decoded_image, add_alpha)
    
    return rgba_image

def get_image_dataset_from_folder(folder_path):
    """Load all images directly from a folder path."""
    file_patterns = [
        f"{folder_path}/*.png",
        f"{folder_path}/*.jpg",
        f"{folder_path}/*.jpeg"
    ]
    
    ds = tf.data.Dataset.list_files(file_patterns, shuffle=True)
    ds = ds.map(lambda x: load_and_preprocess_image(x),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

# -----------------------------------------------------------------------------
# Composite Image Generation
# -----------------------------------------------------------------------------
def apply_composite(overlay=False, 
                   background_images=None, 
                   bg_widths=None, 
                   bg_placement_areas=None,
                   category_iterators=None,
                   category_class_ids=None,
                   category_weights=None,
                   min_scale_cm=1.0,
                   max_scale_cm=5.0,
                   rotation_range=(-5.0, 5.0),
                   allow_overlap=True,
                   max_overlap_pct=0.3,
                   min_object_distance_cm=0.0,
                   edge_avoidance=0.0,
                   prefer_center=0.0,
                   object_blending_strength=0.5,
                   number_of_objects_per_image=(1, 5),
                   output_height=640,
                   output_width=640,
                   max_labels=20,
                   is_video_background=True):
    """
    Generates a composite image by placing multiple objects onto a randomly selected background.
    Objects are resized, rotated, and (if configured) placed without exceeding an overlap threshold.
    Annotations (bounding boxes) are created for each placed object.
    """
    category_class_ids_tensor = tf.constant(category_class_ids, dtype=tf.int32)
    
    # Select background and compute pixels per cm
    bg_image, bg_width_cm, placement_area, overlay_bg_image = select_random_background(
        overlay, background_images, bg_widths, bg_placement_areas, is_video_background)
    bg_h = tf.shape(bg_image)[0]
    bg_w = tf.shape(bg_image)[1]
    bg_ppcm = tf.cast(bg_w, tf.float32) / bg_width_cm

    # Initialize composite image and mask for overlap checking
    comp_image = bg_image
    mask = tf.zeros((bg_h, bg_w), dtype=tf.int32)

    # Define max_retries outside the loop
    max_retries = tf.constant(10)
    
    # Calculate blending parameters from strength
    blending_params = calculate_blending_params(object_blending_strength)
    
    # Create params dict for subfunctions
    params_dict = {
        'rotation_range': rotation_range,
        'min_scale_cm': min_scale_cm,
        'max_scale_cm': max_scale_cm,
        'allow_overlap': allow_overlap,
        'max_overlap_pct': max_overlap_pct
    }

    # Determine number of objects to place
    num_objects = tf.random.uniform([], number_of_objects_per_image[0],
                                  number_of_objects_per_image[1],
                                  dtype=tf.int32)
    
    # Use TensorArray for structured labels: [exists, xmin, ymin, xmax, ymax, class_id]
    labels = tf.TensorArray(tf.float32, size=max_labels, dynamic_size=False, 
                           element_shape=tf.TensorShape([6]))
    
    # Initialize all labels as zeros (non-existent)
    for i in tf.range(max_labels):
        labels = labels.write(i, tf.zeros([6], dtype=tf.float32))

    # Unpack placement area (expected format: [x_min, y_min, x_max, y_max])
    x_min, y_min, x_max, y_max = tf.unstack(tf.cast(placement_area, tf.int32))
    
    # Apply edge avoidance by shrinking placement area
    if edge_avoidance > 0.0:
        edge_margin_w = tf.cast(tf.cast(x_max - x_min, tf.float32) * edge_avoidance, tf.int32)
        edge_margin_h = tf.cast(tf.cast(y_max - y_min, tf.float32) * edge_avoidance, tf.int32)
        x_min = x_min + edge_margin_w
        x_max = x_max - edge_margin_w
        y_min = y_min + edge_margin_h
        y_max = y_max - edge_margin_h
    
    # Calculate center point for prefer_center feature
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    # Convert min_object_distance to pixels
    min_distance_px = tf.cast(min_object_distance_cm * bg_ppcm, tf.int32)

    # Loop through objects
    obj_idx = tf.constant(0)
    
    def condition(idx, comp_image, mask, labels):
        return tf.less(idx, num_objects)
        
    def body(idx, comp_image, mask, labels):
        # Select category based on weights
        num_categories = len(category_iterators)
        
        if category_weights is not None and len(category_weights) == num_categories:
            # Use tf.random.categorical for weighted selection
            weights_tensor = tf.constant([category_weights], dtype=tf.float32)
            logits = tf.math.log(weights_tensor)
            category_idx = tf.random.categorical(logits, 1)[0, 0]
        else:
            # Uniform random selection
            category_idx = tf.random.uniform([], 0, num_categories, dtype=tf.int32)
        
        # Get object from selected category
        obj_image = category_iterators[category_idx].get_next()
        
        # Get class_id directly from tensor
        class_id = category_class_ids_tensor[category_idx]
        
        obj_image = resize_and_rotate(obj_image, min_scale_cm, max_scale_cm, bg_ppcm, params_dict)

        # Initialize placement variables for this object
        placed = tf.constant(False)
        retry_count = tf.constant(0)
        x_coord = tf.constant(0, dtype=tf.int32)
        y_coord = tf.constant(0, dtype=tf.int32)

        # Try to place the object
        def placement_condition(placed, retry_count, x_coord, y_coord):
            return tf.logical_and(tf.logical_not(placed), retry_count < max_retries)
        
        @tf.function
        def check_overlap_tf_safe(mask, obj_image, x, y, max_overlap_pct):
            """Checks if placing the object at (x, y) would exceed allowed overlap, supporting partial visibility."""
            bg_h = tf.shape(mask)[0]
            bg_w = tf.shape(mask)[1]
            obj_h = tf.shape(obj_image)[0]
            obj_w = tf.shape(obj_image)[1]
            
            # Calculate valid regions for the object and background
            obj_x_start = tf.maximum(0, -x)
            obj_x_end = tf.minimum(obj_w, bg_w - x)
            bg_x_start = tf.maximum(0, x)
            bg_x_end = tf.minimum(bg_w, x + obj_w)
            
            obj_y_start = tf.maximum(0, -y)
            obj_y_end = tf.minimum(obj_h, bg_h - y)
            bg_y_start = tf.maximum(0, y)
            bg_y_end = tf.minimum(bg_h, y + obj_h)
            
            # Check if any part of the object is visible
            is_visible = tf.logical_and(
                tf.less(obj_x_start, obj_x_end),
                tf.less(obj_y_start, obj_y_end)
            )
            
            def do_check():
                # Extract visible part of the object
                obj_visible = obj_image[obj_y_start:obj_y_end, obj_x_start:obj_x_end, :]
                
                # Use alpha channel to determine object presence
                alpha_threshold = 25
                obj_mask = tf.cast(obj_visible[..., 3] > alpha_threshold, tf.int32)
                
                # Extract the corresponding region from the background mask
                mask_region = mask[bg_y_start:bg_y_end, bg_x_start:bg_x_end]
                
                # Check overlap
                overlap_pixels = tf.reduce_sum(tf.cast((mask_region + obj_mask) > 1, tf.float32))
                total_obj_pixels = tf.reduce_sum(tf.cast(obj_mask > 0, tf.float32))
                
                overlap_percentage = tf.cond(
                    total_obj_pixels > 0,
                    lambda: overlap_pixels / total_obj_pixels * 100.0,
                    lambda: 0.0
                )
                
                return overlap_percentage <= max_overlap_pct
            
            return tf.cond(is_visible, do_check, lambda: tf.constant(False))

        def placement_body(placed, retry_count, x_coord, y_coord):
            obj_shape = tf.shape(obj_image)
            obj_h = obj_shape[0]
            obj_w = obj_shape[1]
            
            # Calculate object center offsets
            half_w = tf.cast(obj_w // 2, tf.int32)
            half_h = tf.cast(obj_h // 2, tf.int32)
            
            # Generate random center coordinates with prefer_center bias
            if prefer_center > 0.0:
                rand_x = tf.random.uniform([], x_min, x_max, dtype=tf.int32)
                rand_y = tf.random.uniform([], y_min, y_max, dtype=tf.int32)
                
                biased_x = tf.cast(
                    tf.cast(rand_x, tf.float32) * (1.0 - prefer_center) + 
                    tf.cast(center_x, tf.float32) * prefer_center, 
                    tf.int32
                )
                biased_y = tf.cast(
                    tf.cast(rand_y, tf.float32) * (1.0 - prefer_center) + 
                    tf.cast(center_y, tf.float32) * prefer_center, 
                    tf.int32
                )
                center_x_pos = biased_x
                center_y_pos = biased_y
            else:
                center_x_pos = tf.random.uniform([], x_min, x_max, dtype=tf.int32)
                center_y_pos = tf.random.uniform([], y_min, y_max, dtype=tf.int32)
            
            # Convert center coordinates to top-left corner coordinates
            new_x = center_x_pos - half_w
            new_y = center_y_pos - half_h
            
            # Check if any part of the object would be visible
            is_visible = tf.logical_and(
                tf.logical_and(new_x < bg_w, new_y < bg_h),
                tf.logical_and(new_x + obj_w > 0, new_y + obj_h > 0)
            )
            
            new_x = tf.cond(is_visible, lambda: new_x, lambda: x_coord)
            new_y = tf.cond(is_visible, lambda: new_y, lambda: y_coord)
            
            valid = is_visible
            
            # Check minimum distance and overlap
            if min_distance_px > 0:
                effective_overlap = max_overlap_pct - (tf.cast(min_distance_px, tf.float32) / 100.0)
                effective_overlap = tf.maximum(effective_overlap, 0.0)
            else:
                effective_overlap = max_overlap_pct
            
            if not allow_overlap:
                valid = tf.cond(
                    is_visible, 
                    lambda: check_overlap_tf_safe(mask, obj_image, new_x, new_y, effective_overlap),
                    lambda: tf.constant(False)
                )
            
            placed = valid
            return placed, retry_count + 1, new_x, new_y

        # Run the placement loop
        placed, retry_count, x_coord, y_coord = tf.while_loop(
            placement_condition,
            placement_body,
            loop_vars=(placed, retry_count, x_coord, y_coord)
        )

        # Update the image and labels
        def update_fn():
            new_mask = update_mask_matrix(mask, obj_image, x_coord, y_coord)
            new_image = place_object_with_blending(
                comp_image, obj_image, x_coord, y_coord, 
                blur_radius=blending_params['blur_radius'],
                blur_iterations=blending_params['blur_iterations'],
                alpha_threshold=blending_params['alpha_threshold'],
                min_alpha=blending_params['min_alpha']
            )
            
            # Calculate visible bounding box
            obj_h = tf.shape(obj_image)[0]
            obj_w = tf.shape(obj_image)[1]
            bg_h = tf.shape(comp_image)[0]
            bg_w = tf.shape(comp_image)[1]
            
            # Calculate actual visible area (clipped to image boundaries)
            xmin = tf.cast(tf.maximum(x_coord, 0), tf.float32)
            ymin = tf.cast(tf.maximum(y_coord, 0), tf.float32)
            xmax = tf.cast(tf.minimum(x_coord + obj_w, bg_w), tf.float32)
            ymax = tf.cast(tf.minimum(y_coord + obj_h, bg_h), tf.float32)
            
            # Create label: [exists, xmin, ymin, xmax, ymax, class_id]
            label = tf.stack([1.0, xmin, ymin, xmax, ymax, tf.cast(class_id, tf.float32)])
            
            return new_mask, new_image, label

        def no_update_fn():
            return mask, comp_image, tf.zeros([6], dtype=tf.float32)

        new_mask, new_image, label = tf.cond(placed, update_fn, no_update_fn)
        new_labels = labels.write(idx, label)
        
        return idx + 1, new_image, new_mask, new_labels

    # Run the main loop
    _, comp_image, mask, labels = tf.while_loop(
        condition,
        body,
        loop_vars=(obj_idx, comp_image, mask, labels)
    )

    # Convert TensorArray to tensor
    structured_labels = labels.stack()
    
    # Store original dimensions for scaling
    original_height = tf.cast(bg_h, tf.float32)
    original_width = tf.cast(bg_w, tf.float32)
    
    # Resize composite image to output dimensions
    comp_image_resized = tf.image.resize(comp_image, [output_height, output_width])
    comp_image_resized = tf.cast(comp_image_resized, tf.uint8)
    
    # Calculate scale factors
    scale_y = tf.cast(output_height, tf.float32) / original_height
    scale_x = tf.cast(output_width, tf.float32) / original_width
    
    # Scale bounding box coordinates in labels
    # Label format: [exists, xmin, ymin, xmax, ymax, class_id]
    def scale_label(label):
        exists = label[0]
        xmin = label[1] * scale_x
        ymin = label[2] * scale_y
        xmax = label[3] * scale_x
        ymax = label[4] * scale_y
        class_id = label[5]
        return tf.stack([exists, xmin, ymin, xmax, ymax, class_id])
    
    scaled_labels = tf.map_fn(scale_label, structured_labels, dtype=tf.float32)
    
    # Resize overlay image if present
    if overlay_bg_image is not None:
        overlay_bg_image_resized = tf.image.resize(overlay_bg_image, [output_height, output_width])
        overlay_bg_image_resized = tf.cast(overlay_bg_image_resized, tf.uint8)
    else:
        overlay_bg_image_resized = None
    
    return comp_image_resized, scaled_labels, overlay_bg_image_resized


def create_dataset_from_generator(overlay=False,
                                 params=None,
                                 background_images=None,
                                 bg_widths=None,
                                 bg_placement_areas=None,
                                 category_datasets=None,
                                 category_class_ids=None,
                                 num_samples=100):
    """
    Creates a TensorFlow dataset by using Python generator.
    """
    # Create iterators once for each category
    category_iterators = [iter(ds.repeat()) for ds in category_datasets]
    
    def generator():
        """Python generator that yields samples one at a time."""
        for _ in range(num_samples):
            comp_image, structured_labels, overlay_img = apply_composite_eager(
                overlay=overlay,
                background_images=background_images,
                bg_widths=bg_widths,
                bg_placement_areas=bg_placement_areas,
                category_iterators=category_iterators,
                category_class_ids=category_class_ids,
                category_weights=params.get("category_weights", None),
                min_scale_cm=params["min_scale_cm"],
                max_scale_cm=params["max_scale_cm"],
                rotation_range=params["rotation_range"],
                allow_overlap=params["allow_overlap"],
                max_overlap_pct=params["max_overlap_pct"],
                min_object_distance_cm=params.get("min_object_distance_cm", 0.0),
                edge_avoidance=params.get("edge_avoidance", 0.0),
                prefer_center=params.get("prefer_center", 0.0),
                object_blending_strength=params.get("object_blending_strength", 0.5),
                number_of_objects_per_image=params["number_of_objects_per_image"],
                output_height=params["output_height"],
                output_width=params["output_width"],
                max_labels=params["max_labels"],
                is_video_background=params.get("is_video_background", True)
            )
            
            # Image and labels are already resized and scaled in apply_composite
            # Just normalize to [0, 1] range
            fixed_image = tf.cast(comp_image, tf.float32) / 255.0

            if overlay and overlay_img is not None:
                overlay_normalized = tf.cast(overlay_img, tf.float32) / 255.0
                fixed_image = tf.stack([overlay_normalized, fixed_image], axis=0)
            
            # Labels are already scaled to output dimensions
            yield fixed_image.numpy(), structured_labels.numpy()
    
    # Determine output shapes
    if overlay:
        image_shape = (2, params["output_height"], params["output_width"], 3)
    else:
        image_shape = (params["output_height"], params["output_width"], 3)
    
    label_shape = (params["max_labels"], 6)
    
    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=image_shape, dtype=tf.float32),
            tf.TensorSpec(shape=label_shape, dtype=tf.float32)
        )
    )
    
    return dataset


def apply_composite_eager(overlay=False, 
                          background_images=None, 
                          bg_widths=None, 
                          bg_placement_areas=None,
                          category_iterators=None,
                          category_class_ids=None,
                          category_weights=None,
                          min_scale_cm=1.0,
                          max_scale_cm=5.0,
                          rotation_range=(-5.0, 5.0),
                          allow_overlap=True,
                          max_overlap_pct=0.3,
                          min_object_distance_cm=0.0,
                          edge_avoidance=0.0,
                          prefer_center=0.0,
                          object_blending_strength=0.5,
                          number_of_objects_per_image=(1, 5),
                          output_height=640,
                          output_width=640,
                          max_labels=20,
                          is_video_background=True):
    """
    Eager execution version of apply_composite - no @tf.function decorator.
    This allows direct use of iterators without graph compilation issues.
    """
    return apply_composite(
        overlay=overlay,
        background_images=background_images,
        bg_widths=bg_widths,
        bg_placement_areas=bg_placement_areas,
        category_iterators=category_iterators,
        category_class_ids=category_class_ids,
        category_weights=category_weights,
        min_scale_cm=min_scale_cm,
        max_scale_cm=max_scale_cm,
        rotation_range=rotation_range,
        allow_overlap=allow_overlap,
        max_overlap_pct=max_overlap_pct,
        min_object_distance_cm=min_object_distance_cm,
        edge_avoidance=edge_avoidance,
        prefer_center=prefer_center,
        object_blending_strength=object_blending_strength,
        number_of_objects_per_image=number_of_objects_per_image,
        output_height=output_height,
        output_width=output_width,
        max_labels=max_labels,
        is_video_background=is_video_background
    )