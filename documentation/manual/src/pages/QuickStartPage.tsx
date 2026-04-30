import { Link } from 'react-router-dom'
import { quickStartContent } from '../data/siteContent'
import PageHomeLink from '../components/PageHomeLink'
import CopyableCodeBlock, { ImageDisplay } from '../components/CopyableCodeBlock'
import FlowDiagram from '../components/FlowDiagram'

export default function QuickStartPage() {
  const renderSectionBody = (raw: string) => {
    console.log('SECTION BODY RAW (QuickStart):', JSON.stringify(raw))
    const placeholder = '___LINE_BREAK_PLACEHOLDER___'
    // Treat literal "\\n" markers and double-blank lines as manual breaks
    const withLiteralMarkers = raw.replace(/\\n/g, placeholder)
    const withDoubleNewlines = withLiteralMarkers.replace(/(\r?\n){2,}/g, placeholder)
    // Collapse any remaining single newlines into spaces
    const collapsed = withDoubleNewlines.replace(/\r?\n+/g, ' ')
    const parts = collapsed.split(placeholder)
    return parts.flatMap((part, idx) => (idx === 0 ? [part] : [<br key={idx} />, part]))
  }
  return (
    <section className="doc-section">
      <PageHomeLink />
      <h2>{quickStartContent.title}</h2>
      <p>{quickStartContent.description}</p>
      <FlowDiagram title={quickStartContent.title} steps={quickStartContent.stages} />
      <ul className="subsection-link-list">
                <li>
          <Link to="/synthetic-data">Synthetic Data</Link>
        </li>
        <li>
          <Link to="/roboflow-workflow">Roboflow Workflow</Link>
        </li>
        <li>
          <Link to="/training-yolo-model">Training YOLO Model</Link>
        </li>

        <li>
          <Link to="/pose-estimation">Pose Estimation</Link>
        </li>
      </ul>
      <div className="step-grid">
        {quickStartContent.sections.map((section) => (
          <article className="step-card" key={section.heading}>
            <ImageDisplay image={section.image} />
            <h3>{section.heading}</h3>
            <p className="section-body">{renderSectionBody(section.body)}</p>
            <CopyableCodeBlock code={section.code} />
          </article>
        ))}
      </div>
    </section>
  )
}
