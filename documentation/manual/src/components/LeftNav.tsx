import { useState, type MouseEvent as ReactMouseEvent } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import type { NavItem } from '../data/docsContent'

type LeftNavProps = {
  navItems: NavItem[]
  primaryPaths: string[]
  siteHeader: {
    kicker: string
    title: string
    badges: string[]
  }
  isCollapsed: boolean
  onToggleCollapsed: () => void
  onResizeStart: (event: ReactMouseEvent<HTMLElement>) => void
}

export default function LeftNav({
  navItems,
  primaryPaths,
  siteHeader,
  isCollapsed,
  onToggleCollapsed,
  onResizeStart,
}: LeftNavProps) {
  const location = useLocation()
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({})

  const toggleGroup = (groupKey: string) => {
    setExpandedGroups((prev) => ({ ...prev, [groupKey]: !prev[groupKey] }))
  }

  const isGroupOpen = (sectionTo: string, childPaths: string[] = []) => {
    const isParentRoute =
      location.pathname === sectionTo || location.pathname.startsWith(`${sectionTo}/`)
    const isChildRoute = childPaths.some((childPath) =>
      location.pathname.startsWith(childPath),
    )
    return expandedGroups[sectionTo] || isParentRoute || isChildRoute
  }

  return (
    <aside className={`sidebar ${isCollapsed ? 'collapsed' : ''}`} aria-label="section navigation">
      <button
        type="button"
        className="sidebar-collapse-btn"
        onClick={onToggleCollapsed}
        aria-label={isCollapsed ? 'Expand left navigation' : 'Collapse left navigation'}
        title={isCollapsed ? 'Expand navigation' : 'Collapse navigation'}
      >
        {isCollapsed ? '›' : '‹'}
      </button>

      <div className="sidebar-brand">
        {!isCollapsed && (
          <>
            <p className="project-kicker">{siteHeader.kicker}</p>
            <p className="sidebar-brand-title">{siteHeader.title}</p>
            <div className="header-badges" role="list" aria-label="documentation badges">
              {siteHeader.badges.map((badge) => (
                <span key={badge} role="listitem">
                  {badge}
                </span>
              ))}
            </div>
          </>
        )}
      </div>

      {!isCollapsed && <div className="sidebar-scroll">
        <p className="sidebar-title">Documentation Pages</p>
        <nav>
          <ul>
            {navItems.map((section) => (
              <li key={section.to} className="sidebar-group">
                <div className="sidebar-parent-row">
                  <NavLink
                    to={section.to}
                    className={({ isActive }) =>
                      isActive ? 'sidebar-link active' : 'sidebar-link'
                    }
                  >
                    {section.label}
                  </NavLink>
                  {section.children && (
                    <button
                      type="button"
                      className="sidebar-toggle"
                      aria-label={`Toggle ${section.label} subsections`}
                      aria-expanded={isGroupOpen(
                        section.to,
                        section.children.map((child) => child.to),
                      )}
                      onClick={() => toggleGroup(section.to)}
                    >
                      <span
                        className={
                          isGroupOpen(
                            section.to,
                            section.children.map((child) => child.to),
                          )
                            ? 'chevron expanded'
                            : 'chevron'
                        }
                      >
                        ▾
                      </span>
                    </button>
                  )}
                </div>
                {section.children &&
                  isGroupOpen(
                    section.to,
                    section.children.map((child) => child.to),
                  ) && (
                    <ul className="sidebar-subnav">
                      {section.children.map((child) => (
                        <li key={child.to}>
                          <NavLink
                            to={child.to}
                            className={({ isActive }) =>
                              isActive
                                ? 'sidebar-link sidebar-sublink active'
                                : 'sidebar-link sidebar-sublink'
                            }
                          >
                            {child.label}
                          </NavLink>
                        </li>
                      ))}
                    </ul>
                  )}
              </li>
            ))}
          </ul>
        </nav>

        <div className="meta-card">
          <h3>Primary Code Paths</h3>
          {primaryPaths.map((path) => (
            <p key={path}>{path}</p>
          ))}
        </div>
      </div>}

      {!isCollapsed && (
        <div
          className="sidebar-resizer"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize left navigation"
          onMouseDown={onResizeStart}
        />
      )}
    </aside>
  )
}
