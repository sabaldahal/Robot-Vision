export const slugifyApiSegment = (value: string) =>
  value
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1-$2')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')

export const buildApiClassRoute = (moduleBasePath: string, className: string) =>
  `${moduleBasePath}/${slugifyApiSegment(className)}`

export const buildApiMethodId = (className: string, methodName: string) =>
  `${slugifyApiSegment(className)}-${slugifyApiSegment(methodName)}`

export const buildApiMemberId = (className: string, memberName: string) =>
  `${slugifyApiSegment(className)}-${slugifyApiSegment(memberName)}`

export const buildApiMethodRoute = (
  moduleBasePath: string,
  className: string,
  methodName: string,
) => `${buildApiClassRoute(moduleBasePath, className)}#${buildApiMethodId(className, methodName)}`

export const buildApiMemberRoute = (
  moduleBasePath: string,
  className: string,
  memberName: string,
) => `${buildApiClassRoute(moduleBasePath, className)}#${buildApiMemberId(className, memberName)}`