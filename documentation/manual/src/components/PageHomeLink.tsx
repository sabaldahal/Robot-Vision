import { Link } from 'react-router-dom'

export default function PageHomeLink() {
  return (
    <div className="doc-top-actions">
      <Link to="/" className="home-link" aria-label="Go to home">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M3 10.5L12 3L21 10.5"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M6.75 9.75V20.25H17.25V9.75"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </Link>
    </div>
  )
}
