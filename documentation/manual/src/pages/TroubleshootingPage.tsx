import { troubleshootingContent } from '../data/siteContent'
import PageHomeLink from '../components/PageHomeLink'

export default function TroubleshootingPage() {
  return (
    <section className="doc-section">
      <PageHomeLink />
      <h2>{troubleshootingContent.title}</h2>
      {troubleshootingContent.items.map((item) => (
        <details key={item.title}>
          <summary>{item.title}</summary>
          <p>{item.description}</p>
        </details>
      ))}
    </section>
  )
}
