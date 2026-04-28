import { Link } from 'react-router-dom'
import { quickStartContent } from '../data/siteContent'
import PageHomeLink from '../components/PageHomeLink'
import CopyableCodeBlock, { ImageDisplay } from '../components/CopyableCodeBlock'

export default function QuickStartPage() {
  return (
    <section className="doc-section">
      <PageHomeLink />
      <h2>{quickStartContent.title}</h2>
      <p>{quickStartContent.description}</p>
      {/* <CopyableCodeBlock code={quickStartContent.introCode} /> */}
      <div className="architecture-strip">
        {quickStartContent.stages.map((stage) => (
          <span key={stage}>{stage}</span>
        ))}
      </div>
      <ul className="subsection-link-list">
        <li>
          <Link to="/roboflow-workflow">Roboflow Workflow</Link>
        </li>
        <li>
          <Link to="/training-yolo-model">Training YOLO Model</Link>
        </li>
        <li>
          <Link to="/synthetic-data">Synthetic Data</Link>
        </li>
        <li>
          <Link to="/pose-estimation">Pose Estimation</Link>
        </li>
      </ul>
      <div className="step-grid">
        {quickStartContent.sections.map((section) => (
          <article className="step-card" key={section.heading}>
            <ImageDisplay image={section.image} />
            <h3>{section.heading}</h3>
            <p>{section.body}</p>
            <CopyableCodeBlock code={section.code} />
          </article>
        ))}
      </div>
    </section>
  )
}
