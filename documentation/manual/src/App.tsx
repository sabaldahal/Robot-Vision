import './App.css'
import { useEffect, useState, type MouseEvent as ReactMouseEvent } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import LeftNav from './components/LeftNav'
import { navItems } from './data/docsContent'
import { sidebarContent, siteHeader } from './data/siteContent'
import ConfigurationPage from './pages/ConfigurationPage'
import StepDetailPage from './pages/StepDetailPage'
import LandingPage from './pages/LandingPage'
import OverviewPage from './pages/OverviewPage'
import OutputsPage from './pages/OutputsPage'
import PoseEstimationPage from './pages/PoseEstimationPage'
import QuickStartPage from './pages/QuickStartPage'
import RoboflowWorkflowPage from './pages/RoboflowWorkflowPage'
import SyntheticDataPage from './pages/SyntheticDataPage'
import TrainingYOLOModelPage from './pages/Training'
import TroubleshootingPage from './pages/TroubleshootingPage'
import { poseEstimationAPIClasses } from './data/PoseEstimation/api'
import { syntheticDataAPIClasses } from './data/SyntheticDataGeneration/api'
import { syntheticDataSteps } from './data/SyntheticDataGeneration'
import { estimationSteps } from './data/PoseEstimation'
import APIClassPage from './pages/APIClassPage'
import APIModuleIndexPage from './pages/APIModuleIndexPage'

function App() {
  const location = useLocation()
  const [leftNavCollapsed, setLeftNavCollapsed] = useState(false)
  const [leftNavWidth, setLeftNavWidth] = useState(320)
  const [leftNavMobileOpen, setLeftNavMobileOpen] = useState(false)

  useEffect(() => {
    setLeftNavMobileOpen(false)
  }, [location.pathname])

  const startLeftResize = (event: ReactMouseEvent<HTMLElement>) => {
    if (leftNavCollapsed) {
      return
    }

    event.preventDefault()
    const startX = event.clientX
    const startWidth = leftNavWidth

    const onMouseMove = (moveEvent: MouseEvent) => {
      const next = startWidth + (moveEvent.clientX - startX)
      const clamped = Math.max(240, Math.min(460, next))
      setLeftNavWidth(clamped)
    }

    const onMouseUp = () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
  }

  return (
    <div className="page-wrap">
      <div className="mobile-nav-bar" aria-label="mobile navigation actions">
        <button
          type="button"
          className="mobile-nav-toggle mobile-left-nav-toggle"
          onClick={() => setLeftNavMobileOpen((prev) => !prev)}
          aria-label={leftNavMobileOpen ? 'Close section navigation' : 'Open section navigation'}
          aria-expanded={leftNavMobileOpen}
        >
          <span className="mobile-nav-icon" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span className="mobile-nav-label">Menu</span>
        </button>
      </div>
      {leftNavMobileOpen && (
        <button
          type="button"
          className="mobile-nav-backdrop"
          aria-label="Close section navigation overlay"
          onClick={() => setLeftNavMobileOpen(false)}
        />
      )}
      <div
        className={leftNavMobileOpen ? 'layout-grid mobile-left-nav-open' : 'layout-grid'}
        style={{
          ['--left-sidebar-width' as string]: leftNavCollapsed
            ? '56px'
            : `${leftNavWidth}px`,
        }}
      >
        <LeftNav
          navItems={navItems}
          primaryPaths={sidebarContent.primaryPaths}
          siteHeader={siteHeader}
          isCollapsed={leftNavCollapsed}
          mobileOpen={leftNavMobileOpen}
          onToggleCollapsed={() => setLeftNavCollapsed((prev) => !prev)}
          onMobileNavigate={() => setLeftNavMobileOpen(false)}
          onResizeStart={startLeftResize}
        />

        <main className="content" aria-label="documentation content">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/overview" element={<OverviewPage />} />
            <Route path="/quick-start" element={<QuickStartPage />} />

            <Route path="/synthetic-data" element={<SyntheticDataPage />} />
            <Route
              path="/synthetic-data/api"
              element={
                <APIModuleIndexPage
                  module={syntheticDataAPIClasses}
                  moduleBasePath="/synthetic-data/api"
                  overviewTitle="Synthetic Data API Reference"
                  overviewDescription="Dedicated pages for each class used in data generation. Use the right navigation to move across the full reference without losing context."
                />
              }
            />
            {syntheticDataAPIClasses.classes.map((apiClass) => (
              <Route
                key={apiClass.name}
                path={`/synthetic-data/api/${apiClass.path}`}
                element={
                  <APIClassPage
                    module={syntheticDataAPIClasses}
                    moduleBasePath="/synthetic-data/api"
                    overviewPath="/synthetic-data/api"
                    apiClass={apiClass}
                  />
                }
              />
            ))}
            <Route path="/roboflow-workflow" element={<RoboflowWorkflowPage />} />
            <Route path="/training-yolo-model" element={<TrainingYOLOModelPage />} />
            <Route path="/pose-estimation" element={<PoseEstimationPage />} />
            <Route
              path="/pose-estimation/api"
              element={
                <APIModuleIndexPage
                  module={poseEstimationAPIClasses}
                  moduleBasePath="/pose-estimation/api"
                  overviewTitle="Pose Estimation API Reference"
                  overviewDescription="Separate documentation pages for each class involved in detection, pose solving, and evaluation. The right navigation stays consistent across the full reference."
                />
              }
            />
            {poseEstimationAPIClasses.classes.map((apiClass) => (
              <Route
                key={apiClass.name}
                path={`/pose-estimation/api/${apiClass.path}`}
                element={
                  <APIClassPage
                    module={poseEstimationAPIClasses}
                    moduleBasePath="/pose-estimation/api"
                    overviewPath="/pose-estimation/api"
                    apiClass={apiClass}
                  />
                }
              />
            ))}

            {syntheticDataSteps.map((step) => (
              <Route
                key={step.to}
                path={step.to}
                element={
                  <StepDetailPage
                    parentPath="/synthetic-data"
                    parentLabel="Synthetic Data (Blender)"
                    step={step}
                  />
                }
              />
            ))}

            {estimationSteps.map((step) => (
              <Route
                key={step.to}
                path={step.to}
                element={
                  <StepDetailPage
                    parentPath="/pose-estimation"
                    parentLabel="Pose Estimation (YOLO + PnP)"
                    step={step}
                  />
                }
              />
            ))}
            <Route path="/configuration" element={<ConfigurationPage />} />
            <Route path="/outputs" element={<OutputsPage />} />
            <Route path="/troubleshooting" element={<TroubleshootingPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default App
