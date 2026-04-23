import type { APIFunction } from '../data/docsContent'

type APIFunctionCardProps = {
  func: APIFunction
}

export default function APIFunctionCard({ func }: APIFunctionCardProps) {
  return (
    <article className="api-function-card">
      <div className="api-function-header">
        <h3 className="api-function-name">{func.name}</h3>
      </div>

      <p className="api-function-description">{func.description}</p>

      <div className="api-section">
        <h4>Signature</h4>
        <pre className="api-signature">
          <code>{func.signature}</code>
        </pre>
      </div>

      {func.parameters.length > 0 && (
        <div className="api-section">
          <h4>Parameters</h4>
          <table className="api-parameters-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Type</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {func.parameters.map((param) => (
                <tr key={param.name}>
                  <td className="param-name">
                    <code>{param.name}</code>
                    {param.optional && <span className="param-optional">(optional)</span>}
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
            <strong>Type:</strong> <code>{func.returns.type}</code>
          </div>
          <div className="return-description">
            <strong>Description:</strong> {func.returns.description}
          </div>
        </div>
      </div>

      <div className="api-section">
        <h4>Example</h4>
        <pre className="api-example">
          <code>{func.example}</code>
        </pre>
      </div>

      {func.notes && func.notes.length > 0 && (
        <div className="api-section">
          <h4>Notes</h4>
          <ul className="api-notes">
            {func.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </article>
  )
}
