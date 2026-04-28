import PageHomeLink from '../components/PageHomeLink'
import { trainingYOLOModelPageContent } from '../data/siteContent'
import CopyableCodeBlock, { ImageDisplay } from '../components/CopyableCodeBlock'
import FlowDiagram from '../components/FlowDiagram'

export default function TrainingYOLOModelPage() {
	return (
		<section className="doc-section">
			<PageHomeLink />
			<p className="section-kicker">{trainingYOLOModelPageContent.kicker}</p>
			<h2>{trainingYOLOModelPageContent.title}</h2>
			<p>{trainingYOLOModelPageContent.description}</p>
			<FlowDiagram title={trainingYOLOModelPageContent.title} steps={trainingYOLOModelPageContent.stages} />
			<div className="step-grid">
				{trainingYOLOModelPageContent.cards.map((card) => (
					<article className="step-card" key={card.title}>
						<ImageDisplay image={card.image} />
						<h3>{card.title}</h3>
						<p>{card.description}</p>
						{card.file && <code>{card.file}</code>}
						{card.code && <CopyableCodeBlock code={card.code} />}
					</article>
				))}
			</div>
		</section>
	)
}
