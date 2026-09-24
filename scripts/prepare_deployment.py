"""Create an isolated, public-data-only Vercel project from the frozen results."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def prepare():
    source=ROOT/'dashboard'; destination=ROOT/'deploy/vercel'
    destination.mkdir(parents=True,exist_ok=True)
    for directory in ['app','components','lib','tests']:
        shutil.copytree(source/directory,destination/directory,dirs_exist_ok=True)
    component=destination/'components/Dashboard.tsx'
    component.write_text(component.read_text(encoding='utf-8').replace('LOCAL / READ ONLY','PUBLIC / RESEARCH'),encoding='utf-8')
    browser=destination/'tests/browser.mjs'
    browser.write_text(browser.read_text(encoding='utf-8').replace('http://127.0.0.1:3000','http://127.0.0.1:3001'),encoding='utf-8')
    for name in ['package.json','package-lock.json','tsconfig.json','postcss.config.mjs','next.config.ts','next-env.d.ts']:
        if (source/name).exists(): shutil.copy2(source/name,destination/name)
    package=json.loads((destination/'package.json').read_text())
    package['name']='snp-relative-value-public';package['version']='1.1.0'
    package['scripts']['build']='node scripts/build.mjs'
    package['scripts']['start']='next start'
    package['engines']={'node':'22.x'}
    (destination/'package.json').write_text(json.dumps(package,indent=2))
    lock=json.loads((destination/'package-lock.json').read_text())
    lock['name']=package['name'];lock['version']=package['version']
    lock['packages']['']['name']=package['name'];lock['packages']['']['version']=package['version']
    lock['packages']['']['engines']=package['engines']
    (destination/'package-lock.json').write_text(json.dumps(lock,indent=2))
    scripts=destination/'scripts';scripts.mkdir(exist_ok=True)
    (scripts/'build.mjs').write_text("import {spawnSync} from 'node:child_process';\nconst r=spawnSync(process.execPath,['node_modules/next/dist/bin/next','build'],{stdio:'inherit',env:{...process.env,NEXT_PUBLIC_PACKED_DATA:'1',NEXT_TELEMETRY_DISABLED:'1'}});process.exit(r.status??1);\n")
    (destination/'vercel.json').write_text(json.dumps({'framework':'nextjs','buildCommand':'npm run build','installCommand':'npm ci'},indent=2))
    (destination/'.gitignore').write_text('node_modules/\n.next/\n.vercel/\n*.log\n*.tsbuildinfo\n')
    (destination/'.vercelignore').write_text('node_modules/\n.next/\n.git/\ntests/\n*.log\n*.tsbuildinfo\n')
    manifest=[]
    for index,p in enumerate(sorted((source/'public/data').rglob('*.json'))):
        raw=p.read_bytes();value=json.loads(raw)
        if p.parent.name=='snapshots':
            columns=list(value['rows'][0]) if value['rows'] else []
            packed={k:v for k,v in value.items() if k!='rows'}
            packed.update(_format='columnar-snapshot-v1',columns=columns,values=[[row.get(c) for c in columns] for row in value['rows']])
            # Assert exact equality, including all Greeks, outcomes and SHAP values.
            reconstructed={k:v for k,v in packed.items() if k not in ['_format','columns','values']}
            reconstructed['rows']=[dict(zip(columns,r)) for r in packed['values']]
            assert reconstructed==value
        else: packed=value
        encoded=json.dumps(packed,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
        compressed=gzip.compress(encoded,compresslevel=9,mtime=0)
        assert json.loads(gzip.decompress(compressed))==packed
        target=destination/'public/data'/p.relative_to(source/'public/data')
        target=target.with_suffix(target.suffix+'.bin');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(compressed)
        manifest.append({'path':str(target.relative_to(destination)).replace('\\','/'),'source_sha256':hashlib.sha256(raw).hexdigest(),'packed_sha256':hashlib.sha256(compressed).hexdigest(),'source_bytes':len(raw),'packed_bytes':len(compressed)})
        if index%25==0:print(f'Packed {index+1} files',flush=True)
    # The small uncompressed summary supports the existing explicit JSON download.
    summary=json.loads((source/'public/data/summary.json').read_text())
    (destination/'public/data/summary.json').write_text(json.dumps(summary,separators=(',',':')),encoding='utf-8')
    (destination/'DEPLOYMENT_MANIFEST.json').write_text(json.dumps({'version':summary['lock']['config']['version'],'model_sha256':summary['lock']['model_sha256'],'public_access_authorized':True,'lossless':True,'files':manifest,'packed_bytes':sum(r['packed_bytes'] for r in manifest)},indent=2))
    (destination/'README.md').write_text('''# SNP relative-value dashboard

Public deployment of the saved v1.1 research experiment. September is a revised historical evaluation, not a fresh untouched holdout. Results are before costs; source product/Greek conventions remain unverified.

This repository contains the Next.js frontend and losslessly compressed exported research data. It intentionally includes option-level observations and SHAP explanations for public browsing. Raw input files, training caches, Python dependencies, model binaries, private archives, local logs and credentials are not included.

Vercel: import this repository, keep the root directory as `.`, use Node 22 and the checked-in build settings. No environment secrets are required. Build with `npm ci && npm run build`.

The build enables browser-side lossless decompression. All original numeric values and complete snapshot rows are retained. `DEPLOYMENT_MANIFEST.json` records hashes and sizes. Compressed snapshots are fetched only when selected.
''',encoding='utf-8')
    print(json.dumps({'destination':str(destination),'source_data_bytes':sum(r['source_bytes'] for r in manifest),'packed_data_bytes':sum(r['packed_bytes'] for r in manifest),'files':len(manifest)},indent=2))

if __name__=='__main__':prepare()
