import type { APIModule } from '../data/apiContent'
import { buildApiClassRoute, buildApiMethodId, buildApiMemberId, buildApiMethodRoute, buildApiMemberRoute } from '../data/apiRoutes'
import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react'
import { NavLink, Link, useLocation } from 'react-router-dom'

type APIRightNavProps = {
  module: APIModule
  moduleBasePath: string
  isCollapsed: boolean
  onToggleCollapsed: () => void
  onResizeStart: (event: ReactMouseEvent<HTMLElement>) => void
}

export default function APIRightNav({
  module,
  moduleBasePath,
  isCollapsed,
  onToggleCollapsed,
  onResizeStart,
}: APIRightNavProps) {
  const location = useLocation()
  const currentClassName = module.classes.find((apiClass) => {
    const classRoute = buildApiClassRoute(moduleBasePath, apiClass.name)
    return location.pathname === classRoute || location.pathname.startsWith(`${classRoute}/`)
  })?.name

  const [expandedClasses, setExpandedClasses] = useState<Set<string>>(
    new Set(),
  )
  const [activeMethodId, setActiveMethodId] = useState<string>('')
  const [mobileOpen, setMobileOpen] = useState(false)
  const lockedSelectionIdRef = useRef<string>('')
  const lockedSelectionUntilRef = useRef<number>(0)

  const classByMethodId = useMemo(() => {
    const mapping = new Map<string, string>()
    for (const apiClass of module.classes) {
      for (const member of apiClass.members ?? []) {
        mapping.set(buildApiMemberId(apiClass.name, member.name), apiClass.name)
      }
      for (const method of apiClass.methods ?? []) {
        mapping.set(buildApiMethodId(apiClass.name, method.name), apiClass.name)
      }
    }
    return mapping
  }, [module])

  useEffect(() => {
    if (!currentClassName) {
      return
    }

    setExpandedClasses(new Set([currentClassName]))
  }, [currentClassName])

  useEffect(() => {
    if (!location.hash) {
      lockedSelectionIdRef.current = ''
      lockedSelectionUntilRef.current = 0
      setActiveMethodId('')
      return
    }

    const methodId = location.hash.slice(1)
    lockedSelectionIdRef.current = methodId
    lockedSelectionUntilRef.current = Date.now() + 450
    const element = document.getElementById(methodId)
    if (element) {
      element.scrollIntoView({ behavior: 'auto', block: 'start' })
    }
    setActiveMethodId(methodId)
  }, [location.hash, location.pathname])

  useEffect(() => {
    if (!activeMethodId) {
      return
    }

    const activeClassName = classByMethodId.get(activeMethodId)
    if (currentClassName && activeClassName && activeClassName !== currentClassName) {
      setActiveMethodId('')
    }
  }, [activeMethodId, classByMethodId, currentClassName])

  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  useEffect(() => {
    const scrollRoot = document.querySelector('.api-main-content') as HTMLElement | null
    if (!scrollRoot) {
      return
    }

    const methodIds = Array.from(classByMethodId.keys())

    const updateActiveMethodFromScroll = () => {
      if (Date.now() < lockedSelectionUntilRef.current && lockedSelectionIdRef.current) {
        setActiveMethodId((prev) =>
          prev === lockedSelectionIdRef.current ? prev : lockedSelectionIdRef.current,
        )
        return
      }

      const rootRect = scrollRoot.getBoundingClientRect()
      const anchorY = rootRect.top + 120

      let selectedId = ''
      let bestDistance = Number.POSITIVE_INFINITY

      for (const id of methodIds) {
        const methodElement = document.getElementById(id)
        if (!methodElement) {
          continue
        }

        const methodRect = methodElement.getBoundingClientRect()
        const distance = Math.abs(methodRect.top - anchorY)

        if (methodRect.top <= anchorY + 32 && distance < bestDistance) {
          bestDistance = distance
          selectedId = id
        }
      }

      if (!selectedId) {
        for (const id of methodIds) {
          const methodElement = document.getElementById(id)
          if (!methodElement) {
            continue
          }
          const methodRect = methodElement.getBoundingClientRect()
          const distance = Math.abs(methodRect.top - anchorY)
          if (distance < bestDistance) {
            bestDistance = distance
            selectedId = id
          }
        }
      }

      if (!selectedId) {
        return
      }

      setActiveMethodId((prev) => (prev === selectedId ? prev : selectedId))

      const className = classByMethodId.get(selectedId)
      if (className) {
        setExpandedClasses((prev) => {
          const updated = new Set(prev)
          updated.add(className)
          return updated
        })
      }
    }

    updateActiveMethodFromScroll()
    scrollRoot.addEventListener('scroll', updateActiveMethodFromScroll, { passive: true })
    window.addEventListener('resize', updateActiveMethodFromScroll)

    return () => {
      scrollRoot.removeEventListener('scroll', updateActiveMethodFromScroll)
      window.removeEventListener('resize', updateActiveMethodFromScroll)
    }
  }, [classByMethodId, location.pathname])

  const toggleClass = (className: string) => {
    if (expandedClasses.has(className)) {
      setExpandedClasses(new Set())
      return
    }

    setExpandedClasses(new Set([className]))
  }

  return (
    <>
      <button
        type="button"
        className="api-right-nav-mobile-toggle"
        onClick={() => setMobileOpen((prev) => !prev)}
        aria-label={mobileOpen ? 'Close API navigation' : 'Open API navigation'}
        aria-expanded={mobileOpen}
      >
        <span className="mobile-nav-icon" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span className="mobile-nav-label">API</span>
      </button>
      {mobileOpen && (
        <button
          type="button"
          className="mobile-nav-backdrop api-right-nav-backdrop"
          aria-label="Close API navigation overlay"
          onClick={() => setMobileOpen(false)}
        />
      )}
      <nav
        className={`api-right-nav ${isCollapsed ? 'collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`}
      >
      <button
        type="button"
        className="api-right-nav-collapse-btn"
        onClick={onToggleCollapsed}
        aria-label={isCollapsed ? 'Expand API navigation' : 'Collapse API navigation'}
        title={isCollapsed ? 'Expand API navigation' : 'Collapse API navigation'}
      >
        {isCollapsed ? '‹' : '›'}
      </button>

      <div className="api-nav-header">
        <h3>API Reference</h3>
        <p className="api-nav-module">{module.name}</p>
      </div>

      {!isCollapsed && <div className="api-nav-classes">
        {module.classes.map((apiClass) => (
          <div key={apiClass.name} className="api-nav-class">
            <div className="api-nav-class-row">
              <NavLink
                to={buildApiClassRoute(moduleBasePath, apiClass.name)}
                onClick={() => {
                  toggleClass(apiClass.name)
                  setMobileOpen(false)
                }}
                className={({ isActive }) =>
                  isActive || currentClassName === apiClass.name ? 'api-nav-class-link active' : 'api-nav-class-link'
                }
              >
                <span className="class-name">{apiClass.name}</span>
                <span className="api-nav-class-file">{apiClass.file}</span>
              </NavLink>

              <button
                type="button"
                className="api-nav-class-toggle"
                onClick={() => toggleClass(apiClass.name)}
                aria-expanded={expandedClasses.has(apiClass.name)}
                aria-label={`Toggle ${apiClass.name} methods`}
              >
                <span className={expandedClasses.has(apiClass.name) ? 'chevron expanded' : 'chevron'}>
                  ▾
                </span>
              </button>
            </div>

            {expandedClasses.has(apiClass.name) && (
              <ul className="api-nav-methods">
                {apiClass.members && apiClass.members.length > 0 && (
                    <h4>Members</h4>
                )}
                {apiClass.members?.map((member) => {
                    const memberId = buildApiMemberId(apiClass.name, member.name)
                    return (
                      <li key={member.name}>
                        <Link
                            to={buildApiMemberRoute(moduleBasePath, apiClass.name, member.name)}
                            onClick={() => setMobileOpen(false)}
                            className={
                              activeMethodId === memberId ? 'api-nav-method-link active' : 'api-nav-method-link'
                            }
                        >
                          {member.name}
                        </Link>
                      </li>
                    )
                  })}
                {apiClass.methods && apiClass.methods.length > 0 && (
                    <h4>Methods</h4>
                )}
                {apiClass.methods?.map((method) => {
                  const methodId = buildApiMethodId(apiClass.name, method.name)

                  return (
                    <li key={method.name}>
                      <Link
                        to={buildApiMethodRoute(moduleBasePath, apiClass.name, method.name)}
                        onClick={() => setMobileOpen(false)}
                        className={
                          activeMethodId === methodId ? 'api-nav-method-link active' : 'api-nav-method-link'
                        }
                      >
                        {method.name}
                      </Link>
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        ))}
      </div>}

      {!isCollapsed && (
        <div
          className="api-right-nav-resizer"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize API navigation"
          onMouseDown={onResizeStart}
        />
      )}
      </nav>
    </>
  )
}
