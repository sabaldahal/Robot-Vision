// ============ API DOCUMENTATION DATA STRUCTURE ============

export type APIParameter = {
  name: string
  type: string
  description: string
  optional?: boolean
  default?: string
}

export type APIMethod = {
  name: string
  description: string
  signature: string
  parameters: APIParameter[]
  returns: {
    type: string
    description: string
  }
  example: string
  raises?: Array<{
    type: string
    description: string
  }>
  notes?: string[]
}

export type APIClass = {
  path: string,
  name: string
  file: string
  description: string
  methods?: APIMethod[]
  members?: APIParameter[]
}

export type APIModule = {
  id: string
  name: string
  description: string
  classes: APIClass[]
}


