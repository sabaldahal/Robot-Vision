export type Parameter = {
  name: string
  type: string
  description: string
  optional?: boolean
}

export type APIFunction = {
  name: string
  description: string
  signature: string
  parameters: Parameter[]
  returns: {
    type: string
    description: string
  }
  example: string
  notes?: string[]
}

export type DocStep = {
  title: string
  description: string
  file: string
  to: string
  notes: string[]
  sections?: DocStepSection[]
}

export type DocStepSection = {
  heading: string
  body: string
}

type NavChild = {
  to: string
  label: string
}

export type NavItem = {
  to: string
  label: string
  children?: NavChild[]
}

import { syntheticDataSteps } from "./SyntheticDataGeneration"
import { estimationSteps } from "./PoseEstimation"


export const navItems: NavItem[] = [
  { to: '/overview', label: 'Overview' },
  { to: '/quick-start', label: 'Quick Start' },

  {
    to: '/synthetic-data',
    label: 'Synthetic Data (Blender)',
    children: [
      ...syntheticDataSteps.map((step) => ({ to: step.to, label: step.title })),
      { to: '/synthetic-data/api', label: 'API Reference' },
    ],
  },
    { to: '/roboflow-workflow', label: 'Roboflow Workflow' },
  { to: '/training-yolo-model', label: 'Training YOLO Model' },
  {
    to: '/pose-estimation',
    label: 'Pose Estimation (YOLO + PnP)',
    children: [
      ...estimationSteps.map((step) => ({ to: step.to, label: step.title })),
      { to: '/pose-estimation/api', label: 'API Reference' },
    ],
  },
  { to: '/configuration', label: 'Configuration Reference' },
  { to: '/outputs', label: 'Outputs and Artifacts' },
  { to: '/troubleshooting', label: 'Troubleshooting' },
]
