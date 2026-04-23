import { outputsContent } from '../data/siteContent'
import PageHomeLink from '../components/PageHomeLink'

export default function OutputsPage() {
  return (
    <section className="doc-section">
      <PageHomeLink />
      <h2>{outputsContent.title}</h2>
      <div className="artifact-grid">
        {outputsContent.artifacts.map((artifact) => (
          <article key={artifact.title}>
            <h3>{artifact.title}</h3>
            <p>{artifact.description}</p>
          </article>
        ))}
      </div>
    </section>
  )
}
