"""Verify retained package source, evidence and distribution hashes without rerunning simulations."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
receipt=json.loads((root/'evidence/validation_receipt.json').read_text())
errors=[]
for category in ('source_sha256','evidence_sha256','distribution_sha256'):
    for name,expected in receipt[category].items():
        path=root/name
        if category=='distribution_sha256' and not path.exists():
            print(f'Skipping distribution not present locally (rebuild or download it to check): {name}')
            continue
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            errors.append(name)
if errors:
    raise SystemExit('Changed or missing files: '+', '.join(errors))
print('All retained source, evidence and distribution hashes match the receipt.')
