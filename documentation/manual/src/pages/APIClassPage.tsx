import { useEffect, useState, type MouseEvent as ReactMouseEvent } from 'react'
import { Link, useLocation } from 'react-router-dom'
import PageHomeLink from '../components/PageHomeLink'
import APIMobileOutline from '../components/APIMobileOutline'
import APIRightNav from '../components/APIRightNav'
import APIClassSection from '../components/APIClassSection'
import type { APIClass, APIModule } from '../data/apiContent'

type APIClassPageProps = {
  module: APIModule
  moduleBasePath: string
  overviewPath: string
  apiClass: APIClass
}

export default function APIClassPage({
  module,
  moduleBasePath,
  overviewPath,
  apiClass,
}: APIClassPageProps) {
  const location = useLocation()
  const [rightNavCollapsed, setRightNavCollapsed] = useState(false)
  const [rightNavWidth, setRightNavWidth] = useState(320)

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

  useEffect(() => {
    const scrollRoot = document.querySelector('.api-main-content') as HTMLElement | null

    if (!scrollRoot) {
      return
    }

    if (!location.hash) {
      scrollRoot.scrollTo({ top: 0, behavior: 'auto' })
      return
    }

    const element = document.getElementById(location.hash.slice(1))
    if (element) {
      element.scrollIntoView({ behavior: 'auto', block: 'start' })
    }
  }, [location.hash, location.pathname])

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
          <h1>{module.name}</h1>
          <p className="api-page-description">{module.description}</p>
          <APIMobileOutline module={module} moduleBasePath={moduleBasePath} />
        </section>

        <section className="api-class-page-shell">
          <APIClassSection apiClass={apiClass} moduleId={module.id} />

          <footer className="api-page-footer api-class-footer">
            <Link to={overviewPath} className="back-link">
              Back to API index
            </Link>
          </footer>
        </section>
      </main>
    </div>
  )
}