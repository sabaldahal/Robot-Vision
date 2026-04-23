import { Link, useLocation } from 'react-router-dom'
import PageHomeLink from '../components/PageHomeLink'
import APIMobileOutline from '../components/APIMobileOutline'
import APIRightNav from '../components/APIRightNav'
import type { APIModule } from '../data/apiContent'
import { buildApiClassRoute } from '../data/apiRoutes'
import { useEffect, useState, type MouseEvent as ReactMouseEvent } from 'react'

type APIModuleIndexPageProps = {
  module: APIModule
  moduleBasePath: string
  overviewTitle: string
  overviewDescription: string
}

export default function APIModuleIndexPage({
  module,
  moduleBasePath,
  overviewTitle,
  overviewDescription,
}: APIModuleIndexPageProps) {
  const location = useLocation()
  const [rightNavCollapsed, setRightNavCollapsed] = useState(false)
  const [rightNavWidth, setRightNavWidth] = useState(320)

  useEffect(() => {
    const scrollRoot = document.querySelector('.api-main-content') as HTMLElement | null
    if (!scrollRoot) {
      return
    }

    scrollRoot.scrollTo({ top: 0, behavior: 'auto' })
  }, [location.pathname])

  const startRightResize = (event: ReactMouseEvent<HTMLElement>) => {
    if (rightNavCollapsed) {
      return
    }

    event.preventDefault()
    const startX = event.clientX
    const startWidth = rightNavWidth

    const onMouseMove = (moveEvent: MouseEvent) => {
      const next = startWidth + (startX - moveEvent.clientX)
      const clamped = Math.max(240, Math.min(480, next))
      setRightNavWidth(clamped)
    }

    const onMouseUp = () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
  }

  return (
    <div
      className="api-documentation-layout"
      style={{
        ['--right-api-nav-width' as string]: rightNavCollapsed ? '56px' : `${rightNavWidth}px`,
      }}
    >
      <APIRightNav
        module={module}
        moduleBasePath={moduleBasePath}
        isCollapsed={rightNavCollapsed}
        onToggleCollapsed={() => setRightNavCollapsed((prev) => !prev)}
        onResizeStart={startRightResize}
      />

      <main className="api-main-content">
        <section className="api-page-header">
          <PageHomeLink />
          <p className="section-kicker">API Documentation</p>
          <h1>{overviewTitle}</h1>
          <p className="api-page-description">{overviewDescription}</p>
          <APIMobileOutline module={module} moduleBasePath={moduleBasePath} />
        </section>

        <section className="api-index-intro">
          <div className="api-index-copy">
            <p>
              Select a section below to open a dedicated page. The right navigation stays available on every page so
              you can move between classes and methods without losing context.
            </p>
          </div>

          <div className="api-index-card-grid">
            {module.classes.map((apiClass) => (
              <article key={apiClass.name} className="api-index-card">
                <p className="api-index-card-kicker">{apiClass.file}</p>
                <h2>{apiClass.name}</h2>
                <p>{apiClass.description}</p>
                <div className="api-index-card-meta">
                  <span>{apiClass.methods?.length ?? 0} methods</span>
                  <span>{apiClass.members?.length ?? 0} members</span>
                  <Link to={buildApiClassRoute(moduleBasePath, apiClass.name)}>Open section</Link>
                </div>
              </article>
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}