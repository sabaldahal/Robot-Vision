import { Link } from 'react-router-dom'
import PageHomeLink from '../components/PageHomeLink'
import { estimationSteps } from '../data/PoseEstimation'
import { poseEstimationPageContent } from '../data/siteContent'

export default function PoseEstimationPage() {
  return (
    <section className="doc-section">
      <PageHomeLink />
      <p className="section-kicker">{poseEstimationPageContent.kicker}</p>
      <h2>{poseEstimationPageContent.title}</h2>
      <p>{poseEstimationPageContent.description}</p>
      <ul className="subsection-link-list">
        {estimationSteps.map((step) => (
          <li key={`top-${step.to}`}>
            <Link to={step.to}>{step.title}</Link>
          </li>
        ))}
      </ul>
      <div className="step-grid">
        {estimationSteps.map((step) => (
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
