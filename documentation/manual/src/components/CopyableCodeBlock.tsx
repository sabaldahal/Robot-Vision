import { useState } from 'react'

export function ImageDisplay({ image }: { image: string | string[] | undefined }) {
  if (!image) return null

  const images = Array.isArray(image) ? image : [image]

  return (
    <div className="image-display">
      {images.map((src, idx) => (
        <img
          key={idx}
          src={src}
          alt={`Content image ${idx + 1}`}
          className="image-display-img"
        />
      ))}
    </div>
  )
}

export default function CopyableCodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await navigator.clipboard.writeText(code.trim())
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1400)
    } catch (e) {
      // ignore clipboard errors silently
    }
  }

  return (
    <div className={copied ? 'copyable-code-block copied' : 'copyable-code-block'}>
      <div className="copyable-code-actions">
        <button
          type="button"
          className="copyable-code-btn"
          onClick={handleCopy}
          aria-label={copied ? 'Copied' : 'Copy code'}
          title={copied ? 'Copied' : 'Copy code'}
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  )
}

