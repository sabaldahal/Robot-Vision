import type { APIModule } from '../data/apiContent'
import { Link } from 'react-router-dom'
import { buildApiMethodRoute } from '../data/apiRoutes'

type APIMobileOutlineProps = {
  module: APIModule
  moduleBasePath: string
}

export default function APIMobileOutline({ module, moduleBasePath }: APIMobileOutlineProps) {
  return (
    <details className="api-mobile-outline">
      <summary>
        <span>Browse API structure</span>
        <span className="api-mobile-outline-hint">classes and methods</span>
      </summary>

      <div className="api-mobile-outline-body">
        {module.classes.map((apiClass) => (
          <details key={apiClass.name} className="api-mobile-class">
            <summary>
              <span>{apiClass.name}</span>
              <span className="api-mobile-class-file">{apiClass.file}</span>
            </summary>

            <div className="api-mobile-method-links">
              {apiClass.methods?.map((method) => {
                const methodRoute = buildApiMethodRoute(moduleBasePath, apiClass.name, method.name)

                return (
                  <Link key={method.name} to={methodRoute}>
                    {method.name}
                  </Link>
                )
              })}
            </div>
          </details>
        ))}
      </div>
    </details>
  )
}