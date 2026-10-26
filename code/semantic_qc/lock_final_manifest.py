"""Lock the completed age-blinded reviews before linking the independent key."""
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parents[2]
C = R.parent / 'task/eagle_coordination'
D = R / 'code/data_audit/final_semantic_qc_20261026'
D.mkdir(exist_ok=False)
fields = ['qc_id', 'head_correct', 'beak_correct', 'feather_correct', 'foreground_usable', 'orientation_usable', 'reason']
sources = {
    'root192': R / 'code/data_audit/roi_blinded_1143_v2/review_forms/reviewer_root_pages_001_016_completed.csv',
    'paper96': R / 'code/data_audit/roi_blinded_1143_v2/review_forms/reviewer_paper_audit_pages_017_024_final.csv',
    'coordinator855': C / 'full_blind_review_20261026/final_v2_855_with_supplement/coordinator_pages025_096_completed.csv',
    'native101': R / 'code/data_audit/roi_beak_second_review/native_second_review_suggestions.csv',
    'key': R / 'code/data_audit/anonymous_mapping_final.csv',
}
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))
def write(p, rows, columns):
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
for name, path in sources.items():
    shutil.copy2(path, D / (name + '.csv'))
rows = []
for name, n in [('root192', 192), ('paper96', 96), ('coordinator855', 855)]:
    part = read(sources[name])
    assert len(part) == n and list(part[0]) == fields
    rows.extend(part)
assert len(rows) == 1143
byid = {r['qc_id']: dict(r) for r in rows}
assert len(byid) == 1143 and set(byid) == {f'QC{i:04d}' for i in range(1, 1144)}
ledger = []
native = read(sources['native101'])
assert len(native) == len({r['qc_id'] for r in native}) == 101
for suggestion in native:
    assert suggestion['native_image_actually_viewed'] == 'yes'
    r = byid[suggestion['qc_id']]
    assert int(r['qc_id'][2:]) <= 192
    value = suggestion['suggested_beak_correct']
    assert value in {'accept', 'uncertain', 'reject'}
    old = r['beak_correct']
    ledger.append({'qc_id': r['qc_id'], 'field': 'beak_correct', 'before': old, 'after': value,
                   'changed': old != value, 'evidence': suggestion['diagnostic_file'], 'reason': suggestion['reason']})
    r['beak_correct'] = value
    r['reason'] += ' | Native beak adjudication: ' + suggestion['reason']
r = byid['QC0593']
ledger.append({'qc_id': r['qc_id'], 'field': 'foreground_usable', 'before': r['foreground_usable'],
               'after': 'uncertain', 'changed': r['foreground_usable'] != 'uncertain',
               'evidence': 'QC0593_card3x.png', 'reason': 'Root viewed the five-panel enlarged card: a small adjacent-bird fragment remains in the foreground cutout.'})
r['foreground_usable'] = 'uncertain'
r['reason'] += ' | Root enlarged-card adjudication: small adjacent-bird fragment in foreground cutout.'
anonymous = [byid[f'QC{i:04d}'] for i in range(1, 1144)]
for r in anonymous:
    for field in fields[1:-1]:
        assert r[field] in {'accept', 'uncertain', 'reject'}
write(D / 'final_anonymous_review.csv', anonymous, fields)
mapping = read(sources['key'])
links = {r['qc_id']: int(r['image_index']) for r in mapping}
assert len(links) == 1143 and set(links.values()) == set(range(1143))
final = [{'qc_id': r['qc_id'], 'image_index': links[r['qc_id']], **{f: r[f] for f in fields[1:]}} for r in anonymous]
write(D / 'final_manifest.csv', final, ['qc_id', 'image_index', *fields[1:]])
write(D / 'adjudication_ledger.csv', ledger, list(ledger[0]))
proposal = read(C / 'proposed_final_qc_20261026_0832/proposed_anonymous_1143.csv')
proposed = {r['qc_id']: r for r in proposal}
assert all(all(r[f] == proposed[r['qc_id']][f] for f in fields[1:-1]) for r in anonymous)
report = {
    'locked_utc': datetime.now(timezone.utc).isoformat(), 'n_images': 1143,
    'protocol': 'Complete age-blinded visual review, followed by native-mask beak adjudication; independent key linked only after decisions locked. Uncertain/reject only tighten automatic validity; no photos or folds removed.',
    'source_sha256': {k: sha(p) for k, p in sources.items()},
    'native_adjudications': len(native), 'native_counts': dict(Counter(r['suggested_beak_correct'] for r in native)),
    'changed_fields': sum(r['changed'] for r in ledger),
    'final_review_counts': {f: dict(Counter(r[f] for r in anonymous)) for f in fields[1:-1]},
    'independent_anonymous_proposal_decisions_equal': True,
    'output_sha256': {p.name: sha(p) for p in D.iterdir() if p.is_file()},
    'n_unique_qc_ids': len(byid), 'n_unique_image_indices': len(set(links.values())),
    'root_adjudication': 'Adopt all 101 actual-view native suggestions and conservatively tighten QC0593 foreground after root visual inspection.'
}
(D / 'lock_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
