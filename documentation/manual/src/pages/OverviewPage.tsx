import { overviewContent } from '../data/siteContent'
import PageHomeLink from '../components/PageHomeLink'

export default function OverviewPage() {
  return (
    <section className="doc-section">
      <PageHomeLink />
      <h2>{overviewContent.title}</h2>
      <p>{overviewContent.description}</p>
      <div className="architecture-strip">
        {overviewContent.architectureStages.map((stage) => (
          <span key={stage}>{stage}</span>
        ))}
      </div>
    </section>
  )
}
