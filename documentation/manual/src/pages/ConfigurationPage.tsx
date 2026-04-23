import { configurationContent } from '../data/siteContent'
import PageHomeLink from '../components/PageHomeLink'

export default function ConfigurationPage() {
  return (
    <section className="doc-section">
      <PageHomeLink />
      <h2>{configurationContent.title}</h2>
      <div className="table-wrap" role="region" aria-label="configuration table">
        <table>
          <thead>
            <tr>
              <th>Area</th>
              <th>Key Inputs</th>
              <th>Expected Output</th>
            </tr>
          </thead>
          <tbody>
            {configurationContent.rows.map((row) => (
              <tr key={row.area}>
                <td>{row.area}</td>
                <td>{row.keyInputs}</td>
                <td>{row.expectedOutput}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
