"""Workspace convenience launcher. Normal virtual-environment Python also works."""
import os
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
os.chdir(root)
env=os.environ.copy()
deps=root/'.deps'
if deps.exists(): env['PYTHONPATH']=str(deps)+os.pathsep+str(root)
command=sys.argv[1:]
if not command: command=['train']
if command[0] in ['train','evaluate','audit','export_results','preprocess','review_preprocessing']:
    args=[sys.executable,'-m','quant.'+command[0],*command[1:]]
elif command[0]=='test': args=[sys.executable,'-m','pytest','tests','-q',*command[1:]]
elif command[0] in ['dev','build','start','test:ui']:
    node=next((root/'.tools').glob('node-*-win-x64'),None) if (root/'.tools').exists() else None
    if node: env['PATH']=str(node)+os.pathsep+env['PATH']
    env['NEXT_TELEMETRY_DISABLED']='1';env['PLAYWRIGHT_BROWSERS_PATH']=str(root/'.tools/browsers')
    if node: args=[str(node/'node.exe'),str(node/'node_modules/npm/bin/npm-cli.js'),'--prefix','dashboard','run',command[0]]
    else: args=['npm.cmd' if os.name=='nt' else 'npm','--prefix','dashboard','run',command[0]]
else: raise SystemExit('Usage: python scripts/run.py [train|evaluate|audit|preprocess|review_preprocessing|test|dev|build|start|test:ui]')
raise SystemExit(subprocess.call(args,env=env))
