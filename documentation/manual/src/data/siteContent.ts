export const siteHeader = {
  kicker: 'Robot Vision / Technical Manual',
  title: 'Synthetic Data Generation + Pose Estimation Documentation',
  badges: ['Version 1.0', 'Workflow Reference'],
}

export const sidebarContent = {
  primaryPaths: ['data generator/', 'estimator/solver/'],
}

export const overviewContent = {
  title: 'Overview',
  description:
    'This manual explains the operational flow for generating synthetic spacecraft datasets in Blender and estimating 6D pose through YOLO detections and PnP-based geometric solving.',
  architectureStages: [
    'Blender Scene',
    'Annotation Export',
    'YOLO Inference',
    'PnP Solver',
    'Error Metrics',
  ],
}

export const quickStartContent = {
  title: 'Quick Start',
  steps: [
    'Generate synthetic dataset samples with the Blender pipeline.',
    'Train or select YOLO weights from the weights directory.',
    'Run solver inference scripts from estimator/solver.',
    'Evaluate output quality with error_metrics_v3.py.',
  ],
  commandBlock: `
  # Suggested local workflow
  # Make sure Blender is installed and accessible via command line

  cd "data generator"
  blender "path/to/blender_file.blend" --background --python "main.py"

  # Train a YOLO model using the generated dataset
  # (This step can be done separately using a preferred training setup)

  # After obtaining YOLO weights, run inference and pose solving

  cd "../estimator/solver"
  python inference.py
  python pose.py
  `,
}

export const syntheticDataPageContent = {
  kicker: 'Parent Section',
  title: 'Synthetic Data Generation (Blender)',
  description:
    'Browse the subsections below for focused documentation on each stage of the generation pipeline.',
}

export const poseEstimationPageContent = {
  kicker: 'Parent Section',
  title: 'Pose Estimation (YOLO + PnP)',
  description:
    'Each subsection covers one solver stage in depth, from detections to geometric recovery and metric validation.',
}

export const apiReferencePageContent = {
  kicker: 'API Documentation',
  title: 'API Reference',
  description:
    'Detailed function and class documentation for all major modules, including signatures, parameters, return types, and usage examples.',
}

export type ConfigurationRow = {
  area: string
  keyInputs: string
  expectedOutput: string
}

export const configurationContent = {
  title: 'Configuration Reference',
  rows: [
    {
      area: 'Scene Randomization',
      keyInputs: 'Camera bounds, object transforms, lighting distributions',
      expectedOutput: 'Diverse image set for robust detector training',
    },
    {
      area: 'Detection Inference',
      keyInputs: 'Model weights, confidence threshold, image stream',
      expectedOutput: 'Bounding boxes and candidate keypoints',
    },
    {
      area: 'PnP Solve',
      keyInputs: '2D-3D correspondence mapping, camera intrinsics',
      expectedOutput: 'rvec and tvec for each accepted frame',
    },
  ] as ConfigurationRow[],
}

export type OutputArtifact = {
  title: string
  description: string
}

export const outputsContent = {
  title: 'Outputs and Artifacts',
  artifacts: [
    {
      title: 'Generated Dataset',
      description: 'Image frames, labels, and metadata generated from Blender scenes.',
    },
    {
      title: 'Model Weights',
      description: 'Versioned YOLO checkpoints located under the weights directory.',
    },
    {
      title: 'Evaluation Reports',
      description: 'Rotational and translational error report',
    },
  ] as OutputArtifact[],
}

export type TroubleshootingItem = {
  title: string
  description: string
}

export const troubleshootingContent = {
  title: 'Troubleshooting',
  items: [
    {
      title: 'Can\'t understand the model weights format structure',
      description:
        'format_3.x is the current format used for YOLO weights. format_A.x (major version: A in format_A.x) specifies the structure of the keypoints and classes in the dataset while the minor version (x in format_A.x) specifies the training configuration.',
    },

  ] as TroubleshootingItem[],
}
