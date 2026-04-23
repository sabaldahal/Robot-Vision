// ============ SYNTHETIC DATA GENERATION API ============
import type { APIModule } from '../apiContent'

export const syntheticDataAPIClasses: APIModule = {
  id: 'data-generation',
  name: 'Data Generation Module',
  description: 'Core utilities for synthetic dataset generation including bounding box projection, keypoint extraction, and annotation formatting.',
  classes: [
    {
      path: 'config',
      name: 'Config',
      file: 'data generator/utils/config.py',
      description: 'Configuration management for synthetic data generation parameters, including randomization settings and generation controls.',
      members:[
        {
          name: "Verbose",
          type: "bool",
          description: "Enable detailed logging of the data generation process for debugging and traceability.",
          default: "False"
        }, 
        {
          name: "image_resolution_x",
          type: "int",
          description: "Horizontal resolution of the generated images in pixels.",
          default: "1280"
        },
        {
          name: "image_resolution_y",
          type: "int",
          description: "Vertical resolution of the generated images in pixels.",
          default: "720"
        },
        {
          name: "total_images_to_generate",
          type: "int",
          description: "Total number of images to generate.",
          default: "10"
        },
        {
          name: "image_start_index",
          type: "int",
          description: "Starting index for naming generated image files.",
          default: "0" 
        },
        {
          name: "save_annotations_file_after_every_n_images",
          type: "int",
          description: "Frequency (in number of images) to save intermediate annotation files during generation.",
          default: "25"
        },
        {
          name: "export_transformation_matrices",
          type: "bool",
          description: "Whether to export per-frame transformation matrices for debugging, testing and reproducibility.",
          default: "True"
        },
        {
          name: "camera_to_object_distance_range",
          type: "Tuple[float, float]",
          description: "Range (min, max) in meters for randomizing camera distance to the target object.",
          default: "None"
        },
        {
          name: "default_position_bounds",
          type: "Dict[str, Tuple[float, float]]",
          description: "Default bounds for randomizing object positions along each axis (x, y, z) in meters.",
          default: "{'x': (-2.0, 2.0), 'y': (-1.215, 1.215), 'z': (0.93, 2.0)}"
        },
        {
          name: "export_path",
          type: "str",
          description: "Base directory path where generated images and annotations will be saved.",
          default: "some/random/path OR empty string"
        }
      ]
    },
    {
      path: 'sdg-data',
      name: 'SDGData',
      file: 'data generator/utils/sdgdata.py',
      description: 'Structured data class for storing references to the Blender scene, camera, collections, and other relevant data used during synthetic data generation.',
      members: [
        {
          name: 'scene',
          type: 'bpy.types.Scene',
          description: 'Reference to the Blender scene object.'
        },
        {
          name: 'camera',
          type: 'bpy.types.Camera',
          description: 'Reference to the Blender camera object.'
        },
        {
          name: 'resx',
          type: 'int',
          description: 'Horizontal resolution of the rendered images in pixels.'
        },
        {
          name: 'resy',
          type: 'int',
          description: 'Vertical resolution of the rendered images in pixels.'
        },
        {
          name: 'all_classes_collection',
          type: 'bpy.types.Collection[]',
          description: 'Reference to the Blender collection containing all object classes for bounding box projection.'
        },
        {
          name: 'all_keypoints_collection',
          type: 'bpy.types.Collection[]',
          description: 'Reference to the Blender collection containing all keypoint objects for annotation.'
        },
        {
          name: 'keypoint_collection',
          type: 'bpy.types.Collection',
          description: 'Reference to the Blender collection containing all_keypoints_collection'
        },
        {
          name: 'top_collection',
          type: 'bpy.types.Collection',
          description: 'Reference to the Blender collection containing all upper faces of the Spacecraft',
        },
        {
          name: 'bottom_collection',
          type: 'bpy.types.Collection',
          description: 'Reference to the Blender collection containing all lower faces of the Spacecraft',
        },
        {
          name: 'obj_controller',
          type: 'bpy.types.Object',
          description: 'Reference to the Blender object used for controlling the target object\'s position and orientation during randomization.',
        },
        {
          name: 'lights',
          type: 'bpy.types.Collection[]',
          description: 'Reference to the Blender light objects in the scene for randomization of lighting conditions.',
        },
      ],
    },
    {
      path: 'bounds',
      name: 'Bounds',
      file: 'data generator/utils/randomizer.py',
      description: 'Data class defining spatial bounds for randomization of object and camera positions during synthetic data generation.',
      members: [
        {
          name: 'X',
          type: 'Tuple[float, float]',
          description: 'Range (min, max) in meters for position along the X axis.',
          default: "None"
        },
        {
          name: 'Y',
          type: 'Tuple[float, float]',
          description: 'Range (min, max) in meters for position along the Y axis.',
          default: "None"
        },
        {
          name: 'Z',
          type: 'Tuple[float, float]',
          description: 'Range (min, max) in meters for position along the Z axis.',
          default: "None"
        }
      ],
      methods: [
        {
          name: '__init__',
          description: 'Initialize the Bounds class with specified ranges for each axis.',
          signature: '__init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float])',
          parameters: [
            {
              name: 'x',
              type: 'Tuple[float, float]',
              description: 'Range (min, max) in meters for position along the X axis.',
              default: "None"
            },
            {
              name: 'y',
              type: 'Tuple[float, float]',
              description: 'Range (min, max) in meters for position along the Y axis.',
              default: "None"
            },
            {
              name: 'z',
              type: 'Tuple[float, float]',
              description: 'Range (min, max) in meters for position along the Z axis.',
              default: "None"
            }
          ],
          returns: {
            type: 'None',
            description: 'Initializes the Bounds instance with provided ranges for each axis',
          },
          example: `bounds = Bounds(x=(-2.0, 2.0), y=(-1.215, 1.215), z=(-0.93, 2.0))`,
        },
        {
          name: 'setDefault',
          description: 'Set default bounds for all axes based on a provided dictionary of axis names from Config to ranges.',
          signature: 'setDefault() -> Bounds',
          parameters: [],
          returns: {
            type: 'Bounds',
            description: 'Sets the bounds for each axis based on Config.default_position_bounds and returns self object instance',
          },
          example: `bounds = Bounds().setDefault()`,
        }
      ]
    },
    {
      path: 'randomizer-settings',
      name: 'RandomizerSettings',
      file: 'data generator/utils/randomizer.py',
      description: 'Data class encapsulating all configurable parameters for the Randomizer, including camera distance range and position randomization bounds.',
      members: [
        {
          name: 'objectBounds',
          type: 'Bounds',
          description: 'Bounds for randomizing object positions along each axis (x, y, z) in meters.',
          default: "None"
        },
        {
          name: 'cameraBounds',
          type: 'Bounds',
          description: 'Bounds for randomizing camera position along each axis (x, y, z) in meters.',
          default: "None"
        },
        {
          name: 'changeObjectPositionX',
          type: 'bool',
          description: 'Whether to apply randomization to the object\'s X position.',
          default: "True"
        },
        {
          name: 'changeObjectPositionY',
          type: 'bool',
          description: 'Whether to apply randomization to the object\'s Y position.',
          default: "True"
        },
        {
          name: 'changeObjectPositionZ',
          type: 'bool',
          description: 'Whether to apply randomization to the object\'s Z position.',
          default: "False"
        },
        {
          name: 'changeCameraPositionX',
          type: 'bool',
          description: 'Whether to apply randomization to the camera\'s X position.',
          default: "True"
        },
        {
          name: 'changeCameraPositionY',
          type: 'bool',
          description: 'Whether to apply randomization to the camera\'s Y position.',
          default: "True"
        },
        {
          name: 'changeCameraPositionZ',
          type: 'bool',
          description: 'Whether to apply randomization to the camera\'s Z position.',
          default: "True"
        },
        {
          name: 'rotateObjectX',
          type: 'bool',
          description: 'Whether to apply random rotation to the object around the X axis.',
          default: "False"
        },
        {
          name: 'rotateObjectY',
          type: 'bool',
          description: 'Whether to apply random rotation to the object around the Y axis.',
          default: "False"
        },
        {
          name: 'rotateObjectZ',
          type: 'bool',
          description: 'Whether to apply random rotation to the object around the Z axis.',
          default: "True"
        },
        {
          name: 'cameraDistance',
          type: 'Tuple[float, float]',
          description: 'Range (min, max) in meters for randomizing camera distance to the target object.',
          default: "None"
        }
      ]
    },
    {
      path: 'randomizer',
      name: 'Randomizer',
      file: 'data generator/utils/randomizer.py',
      description: 'Utility class for applying controlled randomization to object and camera positions, orientations, and other scene parameters during synthetic data generation.',
      members: [
        {
          name: 'settings',
          type: 'RandomizerSettings',
          description: 'Configuration object containing randomization bounds and parameters for position, rotation, and other scene attributes.',
          default: 'RandomizerSettings()\n with objectBounds and cameraBounds set to Config.default_position_bounds',
        },
        {
          name: 'data',
          type: 'SDGData',
          description: 'Structured data object containing references to the Blender scene, camera, collections, and other relevant data for randomization operations.',
        }
      ],
      methods: [
        {
          name: '__init__',
          description: 'Initialize the Randomizer with SDGData and RandomizerSettings.',
          signature: '__init__(self, data: SDGData, settings: RandomizerSettings)',
          parameters: [
            {
              name: 'data',
              type: 'SDGData',
              description: 'Structured data object with scene and collection references for randomization',
            },
            {
              name: 'settings',
              type: 'RandomizerSettings',
              description: 'Configuration object with randomization bounds and parameters',
              default: 'None',
              optional: true
            },
          ],
          returns: {
            type: 'None',
            description: 'Initializes the Randomizer instance with provided data and settings',
          },
          example: `randomizer = Randomizer(sdg_data, randomizer_settings)`,
          notes: [
            'RandomizerSettings can be customized to control the range and distribution of randomization applied to the scene.',
            'SDGData should be properly initialized with references to the Blender scene, camera, and relevant collections before creating the Randomizer instance.',
          ],
        },
        {
          name: 'randomize_camera_object_position',
          description: 'Randomize the object\'s position and orientation and randomize the camera\'s position around the target object while maintaining a specified distance range defined in the settings and ensuring the camera is oriented towards the object.',
          signature: 'randomize_camera_object_position() -> None',
          parameters: [],
          returns: {
            type: 'None',
            description: 'Applies randomization to the camera position and orientation in the Blender scene',
          },
          example: `randomizer.randomize_camera_object_position()`,
          notes: [
            'Randomizes object position and orientation based on settings',
            'Randomizes camera position within specified bounds while keeping it focused on the target object.',
            'Ensures that the camera-object distance remains within the configured range for consistent framing.',
            'Applies random offsets to camera position for increased variability in generated data.',
            'Internally, the method calls _lookAtObject() to orient the camera towards the target object after randomizing positions.',
            'Internally, the method calls _set_min_max_distance() to enforce the camera-object distance constraints defined in the settings.',
          ],
        },
        {
          name: 'randomize_lights',
          description: 'Randomize the properties of light sources in the scene, such as energy, to create varied lighting conditions for synthetic data generation.',
          signature: 'randomize_lights() -> None',
          parameters: [],
          returns: {
            type: 'None',
            description: 'Applies randomization to light properties in the Blender scene',
          },
          example: `randomizer.randomize_lights()`,
          notes: [
            'Randomizes energy levels of lights within a specified range to simulate different lighting conditions.',
            'Can be extended to randomize additional light properties such as position and type if needed.',
          ]
        }
      ]
    },
    {
      path: 'bounding-box',
      name: 'BoundingBox',
      file: 'data generator/utils/bbox.py',
      description: 'Handles 3D-to-2D projection and bounding box calculation for objects in Blender scenes. Supports both single and multi-class object hierarchies.',
      members: [
        {
          name: 'data',
          type: 'SDGData',
          description: 'Structured data object containing references to the Blender scene, camera, resolution, and keypoint collections.',
        }
      ],
      methods: [
        {
          name: '__init__',
          description: 'Initialize the BoundingBox class with scene data including camera, render settings, and object collections.',
          signature: '__init__(self, data)',
          parameters: [
            {
              name: 'data',
              type: 'SDGData',
              description: 'Object containing scene, camera, resolution (resx, resy), and collection references',
            },
          ],
          returns: {
            type: 'None',
            description: 'Initializes instance with provided scene data',
          },
          example: `bbox_handler = BoundingBox(scene_data)
# scene_data should have: scene, camera, resx, resy, all_classes_collection`,
          notes: [
            'Stores reference to scene data for later projection calculations',
            'No validation performed on data object structure',
          ],
        },
//         {
//           name: 'raycast_detect_corners_obj',
//           description: 'Project a single 3D mesh object\'s vertices to 2D screen coordinates using camera raycasting.',
//           signature: 'raycast_detect_corners_obj(self, obj: bpy.types.Object) -> List[Tuple[float, float]]',
//           parameters: [
//             {
//               name: 'obj',
//               type: 'bpy.types.Object',
//               description: 'A Blender mesh object from the scene collection',
//             },
//           ],
//           returns: {
//             type: 'List[Tuple[float, float]]',
//             description: 'List of (x, y) pixel coordinates for visible vertices in screen space',
//           },
//           example: `obj = bpy.data.objects['Spacecraft_Body']
// coords = bbox_handler.raycast_detect_corners_obj(obj)
// # Returns: [(100.5, 200.3), (150.2, 210.1), ...]`,
//           notes: [
//             'Only includes vertices with z > 0 (in front of camera)',
//             'Automatically handles world-to-camera coordinate transformation',
//             'Uses Blender\'s evaluated depsgraph for accurate mesh data',
//             'Coordinates are in image pixel space (0-width, 0-height)',
//           ],
//         },
//         {
//           name: 'raycast_detect_corners_collection_multiclass',
//           description: 'Generate bounding boxes for multiple object classes by processing each collection as a separate entity.',
//           signature: 'raycast_detect_corners_collection_multiclass(self) -> Dict[str, Tuple[float, float, float, float]]',
//           parameters: [],
//           returns: {
//             type: 'Dict[str, Tuple[float, float, float, float]]',
//             description: 'Dictionary mapping collection names to (x_min, y_min, x_max, y_max) bounding boxes in image coordinates',
//           },
//           example: `bboxes = bbox_handler.raycast_detect_corners_collection_multiclass()
// # Output:
// # {
// #   'Body': (50, 100, 450, 600),
// #   'Solar_Panel_Left': (10, 150, 100, 400),
// #   'Solar_Panel_Right': (500, 150, 590, 400)
// # }`,
//           notes: [
//             'Iterates through self.data.all_classes_collection',
//             'Each collection name becomes a class label',
//             'Returns only collections with at least one visible vertex',
//             'Coordinates are in normalized image space',
//           ],
//         },
        {
          name: 'project_bbox_to_2D_from_collection',
          description: 'Public interface method that returns multi-class bounding boxes.',
          signature: 'project_bbox_to_2D_from_collection(self) -> Dict[str, Tuple[float, float, float, float]]',
          parameters: [],
          returns: {
            type: 'Dict[str, Tuple[float, float, float, float]]',
            description: 'Dictionary of class names to bounding box coordinates. Key: collection name, Value: (x_min, y_min, x_max, y_max) of the bounding box in image coordinates',
          },
          example: `bboxes = bbox_handler.project_bbox_to_2D_from_collection()
for class_name, bbox in bboxes.items():
    print(f"{class_name}: {bbox}")`,
          notes: [
            'Wrapper function for raycast_detect_corners_collection_multiclass()',
            'Preferred method for external API usage',
          ],
        },
      ],
    },
    {
      path: 'key-points',
      name: 'KeyPoints',
      file: 'data generator/utils/keypoints.py',
      description: 'Projects 3D keypoint objects to 2D image coordinates with visibility and occlusion detection using raycasting.',
      members: [
        {
          name: 'data',
          type: 'SDGData',
          description: 'Structured data object containing references to the Blender scene, camera, resolution, and keypoint collections.',
        }
      ],
      methods: [
        {
          name: '__init__',
          description: 'Initialize the KeyPoints class with scene data.',
          signature: '__init__(self, data)',
          parameters: [
            {
              name: 'data',
              type: 'SDGData',
              description: 'Object containing scene, camera, resolution, and keypoint collection references',
            },
          ],
          returns: {
            type: 'None',
            description: 'Initializes instance with scene data',
          },
          example: `kpts = KeyPoints(scene_data)`,
          notes: [
            'Stores reference to scene for raycasting operations',
          ],
        },
        {
          name: 'project_keypoints_to_2D_from_collection',
          description: 'Project all keypoint collections to 2D with visibility and occlusion flags.',
          signature: 'project_keypoints_to_2D_from_collection(self) -> Dict[str, List[KeypointData]]',
          parameters: [],
          returns: {
            type: 'Dict[str, {name: str, x: float, y: float, inFrame: bool, occluded: bool}[]]',
            description: 'Dictionary with collection names as keys and lists of keypoint objects as values',
          },
          example: `keypoints_2d = kpts.project_keypoints_to_2D_from_collection()
# Output:
# {
#   'faceA': [
#     {'name': 'faceA_kpt_0', 'x': 234.5, 'y': 456.2, 'inFrame': True, 'occluded': False},
#     {'name': 'faceA_kpt_1', 'x': 245.1, 'y': 467.8, 'inFrame': True, 'occluded': True},
#   ]
# }`,
          notes: [
            'Iterates through self.data.all_keypoints_collection',
            'Applies raycasting for accurate occlusion detection',
            'Keypoints sorted alphabetically by name',
          ],
        },
//         {
//           name: 'project_keypoints_to_2D',
//           description: 'Project a single keypoint collection to 2D with visibility and occlusion detection.',
//           signature: 'project_keypoints_to_2D(self, collection: bpy.types.Collection) -> List[KeypointData]',
//           parameters: [
//             {
//               name: 'collection',
//               type: 'bpy.types.Collection',
//               description: 'Blender collection containing keypoint objects',
//             },
//           ],
//           returns: {
//             type: 'List[KeypointData]',
//             description: 'List of keypoint dictionaries with 2D coordinates and visibility flags',
//           },
//           example: `kpt_collection = bpy.data.collections['Keypoints_Body']
// keypoints = kpts.project_keypoints_to_2D(kpt_collection)`,
//           notes: [
//             'Uses ray-casting from camera to detect occlusions',
//             'Occlusion tolerance: 0.6 cm (6 mm)',
//             'Automatically filters keypoints outside frame bounds',
//             'Sorted by name (alphanumeric)',
//           ],
//         },
      ],
    },
    {
      path: 'transformation-matrix',
      name: 'TransformationMatrix',
      file: 'data generator/utils/transformation_matrix.py',
      description: 'Utility class for calculating and exporting the object\'s transformation matrix with respect to the camera for each frame during synthetic data generation.',
      members: [
        {
          name: 'data',
          type: 'SDGData',
          description: 'Structured data object containing references to the Blender scene, camera, resolution, and collections.',
        }
      ],
      methods: [
        {
          name: 'calculateMatrix',
          description: 'Calculate the 4x4 transformation matrix representing the object\'s position and orientation relative to the camera for the current frame.',
          signature: 'calculateMatrix(self) -> List[List[float]]',
          parameters: [],
          returns: {
            type: 'List[List[float]]',
            description: '4x4 transformation matrix as a list of lists, where each inner list represents a row of the matrix',
          },
          example: `transformation_matrix = TransformationMatrix(scene_data).calculateMatrix()`,
          notes: [
            'Matrix is calculated based on the object\'s world coordinates and the camera\'s position and orientation',
            'Output is a standard 4x4 homogeneous transformation matrix',
          ],
        },
      ],
    },
    {
      path: 'data-formatter',
      name: 'DataFormatter',
      file: 'data generator/utils/dataformatter.py',
      description: 'Formats projected bounding boxes and keypoints into structured annotation dictionaries compatible with common dataset formats like COCO.',
      members: [
        {
          name: 'data',
          type: 'SDGData',
          description: 'Structured data object containing references to the Blender scene, camera, resolution, and collections.',
        }
      ],
      methods: [
        {
          name: '__init__',
          description: 'Initialize the DataFormatter with scene data.',
          signature: '__init__(self, data)',
          parameters: [
            {
              name: 'data',
              type: 'SDGData',
              description: 'Object containing scene, camera, resolution, and collection references',
            },
          ],
          returns: {
            type: 'None',
            description: 'Initializes instance with scene data for annotation formatting',
          },
          example: `formatter = DataFormatter(scene_data)`,
          notes: [
            'Stores reference to scene data for accessing resolution and collection information during formatting',
          ],
        },
        {
          name: 'export_data_COCO',
          description: 'Format projected bounding boxes and keypoints into a COCO-style annotation dictionary.',
          signature: 'export_data_COCO(self, file:str, saveAfterIterations: int) -> Generator[YieldType=None, SendType=(int, [float, float, float, float], Dict[str, Any]), ReturnType=None]',
          parameters: [
            {
              name: 'file',
              type: 'str',
              description: 'Path to the output file where the COCO-style annotations will be saved',
            },
            {
              name: 'saveAfterIterations',
              type: 'int',
              description: 'Number of iterations after which to save the annotations',
            }
          ],
          returns: {
            type: 'Generator[YieldType=None, SendType=(int, [float, float, float, float], Dict[str, Any][]) OR bool, ReturnType=None]',
            description: 'Generator that yields the formatted COCO-style annotations',
          },
example: `
# Initialize 
coco_data_writer = formatter.export_data_COCO('output.json', 10)
next(coco_data_writer)

# During each iteration of data generation:
coco_data_writer.send((image_index, bbox_data, keypoints_data))

# To save annotations after every N iterations, the generator will automatically 
handle saving to the specified file when the iteration count reaches the
saveAfterIterations threshold.

#Finally, to close the generator after all iterations are complete:

coco_data_writer.send(True)

# Output Format:
{
"info": {},
"licenses": {},
"categories": [
    {"id": 0, "name": "SpaceCrafts", "supercategory": "none"},
    {"id": 1,
     "name": "faceA",
     "supercategory": "SpaceCrafts",
     "keypoints": ["kpt_0", "kpt_1", "kpt_2", ...],
     "skeleton": []
     },
    ...
    ],
"images": [
      {
        "id": 0,
        "file_name": "000000.png", 
        "width": 1280, 
        "height": 720
      },
      ...
    ],
"annotations": [
      {
        "id": 0,
        "image_id": 0,
        "category_id": 1,
        "bbox": [x_min, y_min, width, height],
        "area": area,
        "segmentation": [],
        "iscrowd": 0,
        "keypoints": [x, y, visibility, x, y, visibility, ...],
      },
    ...
    ]
}
`,
          notes: [
            'Formats bounding boxes and keypoints into a COCO-style annotation dictionary',
            'Saves annotations to the specified file after every N iterations',
            'Saves Bounding box data as [x_min, y_min, width, height] in COCO format',
            'Keypoints are formatted as a list of [x, y, visibility] for each keypoint, where visibility is 0 (not visible), 1 (occluded), or 2 (visible and not occluded)',
            'Generator allows for incremental sending of annotation data during the data generation loop and handles periodic saving to file',
          ],
        },
        {
          name: 'export_transformation_matrix',
          description: 'Export the camera\'s transformation matrix for the current frame to a file for debugging and reproducibility.',
          signature: 'export_transformation_matrix(self, dir: str, image_index: int, matrix: List[List[float]]) -> None',
          parameters: [
            {
              name: 'dir',
              type: 'str',
              description: 'Directory path where the transformation matrix file will be saved',
            },
            {
              name: 'image_index',
              type: 'int',
              description: 'Index of the current image/frame for naming the output file',
            },
            {
              name: 'matrix',
              type: 'List[List[float]]',
              description: '4x4 transformation matrix representing the camera\'s position and orientation in world space',
            }
          ],
          returns: {
            type: 'None',
            description: 'Saves the transformation matrix to a file in the specified directory with a name based on the image index',
          },
example: `
# Example usage during data generation loop:
transformation_matrix = TransformationMatrix(data).calculateMatrix()  # This function should return the current camera transformation matrix as a 4x4 list of lists
formatter.export_transformation_matrix('transformation_matrices', image_index, transformation_matrix)
`,
          notes: [
            'Saves the camera transformation matrix for each frame to a separate file for debugging and reproducibility',
            'Files are named based on the image index (e.g., transformation_matrices/frame_0001.txt)',
            'Matrix is saved in a human-readable format (e.g., as a text file with rows of values)',
          ],

        }
      ],
    },
  ],
}