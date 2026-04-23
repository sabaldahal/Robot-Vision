import type { APIMethod } from '../data/apiContent'
import { buildApiMethodId } from '../data/apiRoutes'

type APIMethodDetailProps = {
  method: APIMethod
  className: string
}

export default function APIMethodDetail({ method, className }: APIMethodDetailProps) {
  const methodId = buildApiMethodId(className, method.name)

  return (
    <article className="api-method-detail" id={methodId}>
      <div className="api-method-header">
        <h3 className="api-method-name">{method.name}</h3>
        <span className="api-class-context">{className}.</span>
      </div>

      <p className="api-method-description">{method.description}</p>

      <div className="api-section">
        <h4>Signature</h4>
        <pre className="api-signature">
          <code>{method.signature}</code>
        </pre>
      </div>

      {method.parameters.length > 0 && (
        <div className="api-section">
          <h4>Parameters</h4>
          <table className="api-parameters-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {method.parameters.map((param) => (
                <tr key={param.name}>
                  <td className="param-name">
                    <code>{param.name}</code>
                    {param.optional && <span className="param-optional">optional</span>}
                    {param.default && <span className="param-default">={param.default}</span>}
                  </td>
                  <td className="param-type">
                    <code>{param.type}</code>
                  </td>
                  <td className="param-description">{param.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="api-section">
        <h4>Returns</h4>
        <div className="api-return">
          <div className="return-type">
            <strong>Type:</strong> <code>{method.returns.type}</code>
          </div>
          <div className="return-description">{method.returns.description}</div>
        </div>
      </div>

      <div className="api-section">
        <h4>Example</h4>
        <pre className="api-example">
          <code>{method.example}</code>
        </pre>
      </div>

      {method.raises && method.raises.length > 0 && (
        <div className="api-section">
          <h4>Raises</h4>
          <ul className="api-raises">
            {method.raises.map((error) => (
              <li key={error.type}>
                <strong>
                  <code>{error.type}</code>
                </strong>
                : {error.description}
              </li>
            ))}
          </ul>
        </div>
      )}

      {method.notes && method.notes.length > 0 && (
        <div className="api-section">
          <h4>Notes</h4>
          <ul className="api-notes">
            {method.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </article>
  )
}
