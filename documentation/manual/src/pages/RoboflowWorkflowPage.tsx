import PageHomeLink from '../components/PageHomeLink'
import { roboflowWorkflowPageContent } from '../data/siteContent'
import CopyableCodeBlock, { ImageDisplay } from '../components/CopyableCodeBlock'

export default function RoboflowWorkflowPage() {
  return (
    <section className="doc-section">
      <PageHomeLink />
      <p className="section-kicker">{roboflowWorkflowPageContent.kicker}</p>
      <h2>{roboflowWorkflowPageContent.title}</h2>
      <p>{roboflowWorkflowPageContent.description}</p>
      <div className="architecture-strip">
        {roboflowWorkflowPageContent.stages.map((stage) => (
          <span key={stage}>{stage}</span>
        ))}
      </div>
      <div className="step-grid">
        {roboflowWorkflowPageContent.cards.map((card) => (
          <article className="step-card" key={card.title}>
            
            <h3>{card.title}</h3>
            <p>{card.description}</p>
            {card.file && <code>{card.file}</code>}
            {card.code && <CopyableCodeBlock code={card.code} />}
            <ImageDisplay image={card.image} />
          </article>
        ))}
      </div>
    </section>
  )
}