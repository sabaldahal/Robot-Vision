import { overviewContent } from '../data/siteContent'
import PageHomeLink from '../components/PageHomeLink'
import FlowDiagram from '../components/FlowDiagram'

export default function OverviewPage() {
  return (
    <section className="doc-section">
      <PageHomeLink />
      <h2>{overviewContent.title}</h2>
      <p>{overviewContent.description}</p>
      <FlowDiagram title={overviewContent.title} steps={overviewContent.architectureStages} />
    </section>
  )
}
