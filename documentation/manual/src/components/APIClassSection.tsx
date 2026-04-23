import type { APIClass } from '../data/apiContent'
import { buildApiMemberId } from '../data/apiRoutes'
import APIMethodDetail from './APIMethodDetail'

type APIClassSectionProps = {
    apiClass: APIClass
    moduleId: string
}

export default function APIClassSection({ apiClass, moduleId }: APIClassSectionProps) {
    return (
        <section className="api-class-section" id={`${moduleId}-${apiClass.name.toLowerCase()}`}>
            <div className="api-class-header">
                <h2 className="api-class-name">{apiClass.name}</h2>
                <p className="api-class-file">File: {apiClass.file}</p>
            </div>

            <p className="api-class-description">{apiClass.description}</p>

            {apiClass.members && apiClass.members.length > 0 && (
                <div className="api-methods-container">
                <div className="api-section">
                    <h4>Members</h4>
                    <table className="api-parameters-table">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Type</th>
                                <th>Description</th>
                                <th>Default</th>
                            </tr>
                        </thead>
                        <tbody>
                            {apiClass.members?.map((member) => (
                                <tr key={member.name}>
                                    <td className="param-name">
                                        <code>
                                            <span id={buildApiMemberId(apiClass.name, member.name)}>
                                              {member.name}
                                            </span>
                                        </code>
                                    </td>
                                    <td className="param-type">
                                        <code>{member.type}</code>
                                    </td>
                                    <td className="param-description">{member.description}</td>
                                    <td className="param-default">
                                        <code>{member.default ?? 'None'}</code>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                </div>
            )}
            <div className="api-methods-container">
                {apiClass.methods?.map((method) => (
                    <APIMethodDetail key={method.name} method={method} className={apiClass.name} />
                ))}
            </div>
        </section>
    )
}
