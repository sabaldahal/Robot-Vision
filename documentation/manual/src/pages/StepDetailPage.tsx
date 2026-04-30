import { Link } from 'react-router-dom'
import PageHomeLink from '../components/PageHomeLink'
import type { DocStep } from '../data/docsContent'

type StepDetailPageProps = {
  parentPath: string
  parentLabel: string
  step: DocStep
}

export default function StepDetailPage({
  parentPath,
  parentLabel,
  step,
}: StepDetailPageProps) {
  const renderSectionBody = (raw: string) => {
    console.log('SECTION BODY RAW (StepDetail):', JSON.stringify(raw))
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
      <p className="section-kicker">Subsection</p>
      <h2>{step.title}</h2>
      <p>{step.description}</p>

      <div className="detail-meta">
        <span>Primary File</span>
        <code>{step.file}</code>
      </div>

      <div className="detail-panel">
        <h3>Implementation Notes</h3>
        <ul>
          {step.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </div>
      {step.sections && step.sections.map((section) => (
        <div key={section.heading} className="section-panel">
          <h3>{section.heading}</h3>
          <p className='section-body'>{renderSectionBody(section.body)}</p>
        </div>
      ))}

      <Link to={parentPath} className="back-link">
        Back to {parentLabel}
      </Link>
    </section>
  )
}
