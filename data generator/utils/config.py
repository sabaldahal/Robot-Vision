
class Config:
    Verbose = False
    image_resolution_x = 1280
    image_resolution_y = 720
    total_images_to_generate = 2
    image_start_index = 0
    save_annotations_file_after_every_n_images = 25
    export_transformation_matrices = True
    camera_to_object_distance_range = None # (min_distance, max_distance) in metres, set to None to disable
    default_position_bounds = {
        'x': (-2.0, 2.0),
        'y': (-1.215, 1.215),
        'z': (0.93, 2.0)
    }
    export_path = "/Users/sabaldahal/Desktop/College/WORK-RESEARCH LAB/spacecraft blender/src/v2/Robot-Vision/local/test_export_new_config"
