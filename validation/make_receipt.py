"""Regenerate evidence/validation_receipt.json hashes. Run after final build; records execution facts passed in."""
import hashlib, json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
def h(p): return hashlib.sha256((root / p).read_bytes()).hexdigest()
old = json.loads((root / 'evidence/validation_receipt.json').read_text())
sources = sorted(p.relative_to(root).as_posix() for p in
    [*root.glob('*.md'), *root.glob('*.toml'), *root.glob('*.cff'), *root.glob('*.bib'), root / 'LICENSE', root / 'MANIFEST.in', root / '.gitignore',
     *root.glob('docs/*.md'), *root.glob('examples/*'), *root.glob('src/simaudit/**/*.py'), *root.glob('tests/*.py'),
     *root.glob('validation/*'), *root.glob('.github/workflows/*')] if p.is_file() and '__pycache__' not in p.parts)
evidence = sorted(p.relative_to(root).as_posix() for p in root.glob('evidence/*') if p.name != 'validation_receipt.json')
dist = sorted(p.relative_to(root).as_posix() for p in root.glob('dist/*'))
old['source_sha256'] = {p: h(p) for p in sources}
old['evidence_sha256'] = {p: h(p) for p in evidence}
old['distribution_sha256'] = {p: h(p) for p in dist}
old.update(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
(root / 'evidence/validation_receipt.json').write_text(json.dumps(old, indent=2) + '\n')
