"""Reproduce from ONE external data/model/result ZIP; no bundled resources or paid APIs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
from external_resources import restore_resources,sha256

ROOT=Path(__file__).resolve().parents[1]


def locate_resource(root,catalog,provided=None):
    root=Path(root).resolve();name=catalog['archive_name']
    if Path(name).name!=name or '/' in name or '\\' in name:raise RuntimeError('Invalid external archive filename')
    candidates=[Path(provided).resolve()] if provided else [root/name,root.parent/name]
    for path in candidates:
        if path.is_dir():path=path/name
        if path.is_file():return path
    raise RuntimeError(f'找不到外部资源包 {name}。请先下载并放在源码目录或其同级目录，或用 --materials 指定ZIP路径。')


def download_resource(url,target,catalog,*,opener=urllib.request.urlopen):
    if urllib.parse.urlsplit(url).scheme!='https':raise RuntimeError('External download requires HTTPS direct file URL')
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.downloading')
    try:
        total=0
        with opener(url,timeout=45) as source,temporary.open('xb') as output:
            for block in iter(lambda:source.read(1024*1024),b''):
                total+=len(block)
                if total>catalog['archive_bytes']:raise RuntimeError('Downloaded resource exceeds declared size')
                output.write(block)
        if total!=catalog['archive_bytes'] or sha256(temporary)!=catalog['archive_sha256']:
            raise RuntimeError('Downloaded file SHA256/size differs; a cloud sharing webpage is not a direct ZIP download')
        if target.exists():
            if sha256(target)!=catalog['archive_sha256']:raise RuntimeError('Existing downloaded file differs; not overwritten')
        else:temporary.replace(target)
        return target
    finally:
        if temporary.exists():temporary.unlink()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materials','--materials-zip','--materials-dir',dest='materials',type=Path,help='ONE resource ZIP or a directory containing that ZIP')
    parser.add_argument('--url',help='Optional HTTPS direct ZIP URL, not a cloud sharing webpage')
    parser.add_argument('--prepare-only',action='store_true',help='Restore/check inputs without loading scientific model libraries')
    args=parser.parse_args()
    if args.materials is not None and args.url:parser.error('Choose --materials or --url')
    catalog=json.loads((ROOT/'reproducibility/manifest.json').read_text(encoding='utf-8-sig'))
    if catalog.get('schema')!='cluener-external-resources-1':raise RuntimeError('This source ZIP requires the matching NEW single resource ZIP')
    if sha256(ROOT/'reports/demo_inputs_manifest.json')!=catalog['frozen_input_manifest_sha256']:
        raise RuntimeError('Frozen numerical input inventory differs from resource catalog')
    folder=ROOT/'runs/submission_reproduction';folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'process.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode())
        archive=download_resource(args.url,folder/catalog['archive_name'],catalog) if args.url else locate_resource(ROOT,catalog,args.materials)
        print('读取单一外部资源ZIP：'+str(archive),flush=True)
        restored=restore_resources(ROOT,archive,catalog)
        record={'status':'INPUTS_VERIFIED','resource_zip':str(archive),**restored,'scoring_completed':False,
                'project_root':str(ROOT),'new_model_api_calls':0}
        receipt=folder/'verification.json';receipt.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if args.prepare_only:
            print('PASS: SUBMISSION INPUTS VERIFIED; scoring not executed.',flush=True);return
        env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'src')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['PYTHONNOUSERSITE']='1';env['PYTHONDONTWRITEBYTECODE']='1'
        command=[sys.executable,'-B',str(ROOT/'scripts/demo_offline.py'),'--name','源码资源配套复现','--student-id','NOT_APPLICABLE','--auto']
        outcome=subprocess.run(command,cwd=ROOT,env=env)
        if outcome.returncode:raise RuntimeError(f'Scoring exited {outcome.returncode}; resources preserved; no model API requested')
        record.update(status='PASS',scoring_completed=True);receipt.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('PASS: SUBMISSION REPRODUCTION COMPLETE; 15 methods; no model API requests.',flush=True)
    finally:
        os.close(fd);lock.unlink()

if __name__=='__main__':
    try:main()
    except (OSError,RuntimeError,ValueError,KeyError) as error:
        print('源码包复现停止：'+str(error),file=sys.stderr);raise SystemExit(1)
