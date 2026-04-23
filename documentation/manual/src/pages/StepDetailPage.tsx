import { Link } from 'react-router-dom'
import PageHomeLink from '../components/PageHomeLink'
import type { DocStep } from '../data/docsContent'
import parse from 'html-react-parser'

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
          <p className='section-body'>{parse(section.body.replace(/\n(?!\n)/g, ' ').replace(/\\n/g, '\n').trim())}</p>
        </div>
      ))}

      <Link to={parentPath} className="back-link">
        Back to {parentLabel}
      </Link>
    </section>
  )
}
