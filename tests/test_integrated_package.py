"""Packaging must not leak credentials, follow external inputs or accept corrupt evidence."""
import importlib.util
from pathlib import Path
import zipfile
import pytest

spec=importlib.util.spec_from_file_location('integrated_pack',Path(__file__).resolve().parents[1]/'scripts/package_integrated_results.py')
pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)


def test_allowlist_keeps_template_but_excludes_actual_env_and_private_keys():
    assert pack.allowed(Path('.env.example'))
    for name in ['.env','.env.local','key.pem','key.key','key.pfx','process.lock','x/__pycache__/a.pyc','.git/config']:
        assert not pack.allowed(Path(name))


def test_archive_verifies_and_preserves_input(tmp_path):
    root=tmp_path/'project';root.mkdir();p=root/'result.json';p.write_text('{"f1": 0.7}')
    target=pack.build_archive(root,root/'out',[p],[])
    assert p.read_text()=='{"f1": 0.7}'
    assert target.exists()
    with zipfile.ZipFile(target) as z:assert set(z.namelist())=={'result.json','PACKAGE_MANIFEST.json'}
    with pytest.raises(RuntimeError):pack.verify_archive(target,{'result.json':'invalidhash'})


def test_secret_detection_blocks_and_removes_incomplete_archive(tmp_path):
    p=tmp_path/'result.json';p.write_text('accidental_super_secret_key')
    out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='Credential found'):
        pack.build_archive(tmp_path,out,[p],[b'super_secret_key'])
    assert not list(out.glob('*.zip')) and not list(out.glob('*.tmp'))


def test_external_input_is_rejected(tmp_path):
    root=tmp_path/'project';root.mkdir();external=tmp_path/'outside.txt';external.write_text('outside')
    with pytest.raises(RuntimeError,match='Disallowed'):
        pack.build_archive(root,root/'out',[external],[])
