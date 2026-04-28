import { Link } from 'react-router-dom'
import PageHomeLink from '../components/PageHomeLink'
import { landingPageContent, siteHeader } from '../data/siteContent'

export default function LandingPage() {
  return (
    <section className="doc-section landing-page">
      <PageHomeLink />
      <p className="section-kicker">{landingPageContent.kicker}</p>
      <h1 className="landing-title">{landingPageContent.title}</h1>
      <p className="landing-description">{landingPageContent.description}</p>

      <div className="landing-badges" role="list" aria-label="documentation highlights">
        {siteHeader.badges.map((badge) => (
          <span key={badge} role="listitem">
            {badge}
          </span>
        ))}
      </div>

      <div className="landing-card-grid">
        {landingPageContent.actions.map((action) => (
          <Link key={action.to} to={action.to} className="landing-card">
            <h2>{action.title}</h2>
            <p>{action.description}</p>
            <span className="landing-card-cta">Open section</span>
          </Link>
        ))}
      </div>
    </section>
  )
}