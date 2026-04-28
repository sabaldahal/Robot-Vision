type FlowDiagramProps = {
  title: string
  steps: string[]
  className?: string
}

export default function FlowDiagram({ title, steps, className = '' }: FlowDiagramProps) {
  return (
    <section className={`flow-diagram ${className}`.trim()} aria-label={title}>
      <div className="flow-diagram-header">
        <p className="flow-diagram-kicker">Pipeline Flow</p>
        <h3>{title}</h3>
      </div>

      <ol className="flow-diagram-track">
        {steps.map((step, index) => (
          <li key={`${step}-${index}`} className="flow-diagram-node">
            <span className="flow-diagram-index">{index + 1}</span>
            <span className="flow-diagram-text">{step}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}