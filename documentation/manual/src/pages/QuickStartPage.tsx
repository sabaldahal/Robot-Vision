import { quickStartContent } from '../data/siteContent'
import PageHomeLink from '../components/PageHomeLink'

export default function QuickStartPage() {
  return (
    <section className="doc-section">
      <PageHomeLink />
      <h2>{quickStartContent.title}</h2>
      <ol className="numbered">
        {quickStartContent.steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ol>
      <pre>{quickStartContent.commandBlock}</pre>
    </section>
  )
}
