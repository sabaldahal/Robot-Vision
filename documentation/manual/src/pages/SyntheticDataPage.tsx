import { Link } from 'react-router-dom'
import PageHomeLink from '../components/PageHomeLink'
import { syntheticDataSteps } from '../data/SyntheticDataGeneration'
import { syntheticDataPageContent } from '../data/siteContent'

export default function SyntheticDataPage() {
  return (
    <section className="doc-section">
      <PageHomeLink />
      <p className="section-kicker">{syntheticDataPageContent.kicker}</p>
      <h2>{syntheticDataPageContent.title}</h2>
      <p>{syntheticDataPageContent.description}</p>
      <ul className="subsection-link-list">
        {syntheticDataSteps.map((step) => (
          <li key={`top-${step.to}`}>
            <Link to={step.to}>{step.title}</Link>
          </li>
        ))}
      </ul>
      <div className="step-grid">
        {syntheticDataSteps.map((step) => (
          <article className="step-card" key={step.title}>
            <h3>{step.title}</h3>
            <p>{step.description}</p>
            <code>{step.file}</code>
            {/* <ul>
              {step.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul> */}
            <Link to={step.to} className="step-link">
              Open
            </Link>
          </article>
        ))}
      </div>
    </section>
  )
}
