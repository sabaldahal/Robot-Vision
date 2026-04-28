import PageHomeLink from '../components/PageHomeLink'
import { trainingYOLOModelPageContent } from '../data/siteContent'
import CopyableCodeBlock, { ImageDisplay } from '../components/CopyableCodeBlock'

export default function TrainingYOLOModelPage() {
	return (
		<section className="doc-section">
			<PageHomeLink />
			<p className="section-kicker">{trainingYOLOModelPageContent.kicker}</p>
			<h2>{trainingYOLOModelPageContent.title}</h2>
			<p>{trainingYOLOModelPageContent.description}</p>
			<div className="architecture-strip">
				{trainingYOLOModelPageContent.stages.map((stage) => (
					<span key={stage}>{stage}</span>
				))}
			</div>
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
