"""Synthetic tests for a source-disjoint, single-ZIP external resource bundle."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import warnings

ENTRY=Path(__file__).resolve().parents[1]/'scripts/external_resources.py'
class ExternalResourceTests(unittest.TestCase):
 def setUp(self):
  self.assertTrue(ENTRY.exists(),'Missing single-resource ZIP reader')
  spec=importlib.util.spec_from_file_location('external_resources_test',ENTRY);self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name)/'source';self.root.mkdir();self.archive=Path(self.temp.name)/'resource.zip'
 def fixture(self, members=None):
  members=members or {'data/test.jsonl':b'{"synthetic":true}\n','runs/hmm/model.pkl':b'synthetic-not-a-real-model'}
  inventory={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in members.items()}
  raw=json.dumps({'schema':'cluener-resource-members-1','files':inventory}).encode()
  with zipfile.ZipFile(self.archive,'w') as z:
   for n,b in members.items():z.writestr(n,b)
   z.writestr('MATERIALS_MANIFEST.json',raw)
  code=self.root/'src/x.py';code.parent.mkdir();code.write_bytes(b'print(1)')
  frozen={'files_sha256':{'src/x.py':hashlib.sha256(code.read_bytes()).hexdigest()}}
  frozen['files_sha256'].update({n:d['sha256'] for n,d in inventory.items() if n.startswith(('data/','runs/'))})
  manifest=self.root/'reports/demo_inputs_manifest.json';manifest.parent.mkdir();manifest.write_text(json.dumps(frozen))
  return self.catalog(raw,len(inventory)),inventory
 def catalog(self, raw, count):
  body=self.archive.read_bytes()
  return {'schema':'cluener-external-resources-1','archive_name':'resource.zip','archive_bytes':len(body),
    'archive_sha256':hashlib.sha256(body).hexdigest(),'manifest_sha256':hashlib.sha256(raw).hexdigest(),'file_count':count}
 def test_restore_and_no_source_modification(self):
  catalog,inventory=self.fixture();result=self.m.restore_resources(self.root,self.archive,catalog,{'src/x.py'})
  self.assertEqual(result['restored'],2);self.assertEqual((self.root/'src/x.py').read_bytes(),b'print(1)')
  self.assertTrue(result['final_inputs_verified'])
 def test_existing_different_file_is_preserved(self):
  catalog,_=self.fixture();p=self.root/'data/test.jsonl';p.parent.mkdir();p.write_bytes(b'my edit')
  with self.assertRaisesRegex(RuntimeError,'Existing.*differs'):
   self.m.restore_resources(self.root,self.archive,catalog,set())
  self.assertEqual(p.read_bytes(),b'my edit')
 def test_whole_archive_hash_mismatch(self):
  catalog,_=self.fixture();catalog['archive_sha256']='0'*64
  with self.assertRaisesRegex(RuntimeError,'archive SHA256'):
   self.m.restore_resources(self.root,self.archive,catalog,set())
 def test_same_path_in_source_package_is_rejected(self):
  catalog,_=self.fixture()
  with self.assertRaisesRegex(RuntimeError,'overlap'):
   self.m.restore_resources(self.root,self.archive,catalog,{'data/test.jsonl'})
 def test_source_code_inside_resource_is_rejected(self):
  catalog,_=self.fixture({'src/evil.py':b'bad'})
  with self.assertRaisesRegex(RuntimeError,'resource path'):
   self.m.restore_resources(self.root,self.archive,catalog,set())
 def test_missing_frozen_input_is_rejected_before_restore(self):
  catalog,_=self.fixture();p=self.root/'reports/demo_inputs_manifest.json';data=json.loads(p.read_text());data['files_sha256']['runs/missing.json']='0'*64;p.write_text(json.dumps(data))
  with self.assertRaisesRegex(RuntimeError,'Frozen input'):
   self.m.restore_resources(self.root,self.archive,catalog,set())
  self.assertFalse((self.root/'data/test.jsonl').exists())
 def test_duplicate_zip_members_are_rejected(self):
  catalog,_=self.fixture()
  with zipfile.ZipFile(self.archive,'a') as z:
   raw=z.read('MATERIALS_MANIFEST.json')
   with warnings.catch_warnings():warnings.simplefilter('ignore');z.writestr('data/test.jsonl',b'duplicate')
  catalog=self.catalog(raw,2)
  with self.assertRaisesRegex(RuntimeError,'Duplicate'):
   self.m.restore_resources(self.root,self.archive,catalog,set())
 def test_unsafe_paths_are_rejected(self):
  for name in ['../escape','/absolute','C:/escape','data\\escape']:
   with self.subTest(name=name),self.assertRaises(RuntimeError):self.m.safe_target(self.root,name)
 def test_symlink_member_is_rejected(self):
  catalog,_=self.fixture()
  with zipfile.ZipFile(self.archive,'r') as z:items={n:z.read(n) for n in z.namelist()}
  with zipfile.ZipFile(self.archive,'w') as z:
   for name,body in items.items():
    entry=zipfile.ZipInfo(name)
    if name=='data/test.jsonl':entry.create_system=3;entry.external_attr=(0o120777<<16)
    z.writestr(entry,body)
  catalog=self.catalog(items['MATERIALS_MANIFEST.json'],2)
  with self.assertRaisesRegex(RuntimeError,'Symlink'):
   self.m.restore_resources(self.root,self.archive,catalog,set())
 def test_replay_preserves_identical_existing_files(self):
  catalog,_=self.fixture();self.m.restore_resources(self.root,self.archive,catalog,set())
  again=self.m.restore_resources(self.root,self.archive,catalog,set())
  self.assertEqual(again['restored'],0);self.assertEqual(again['preserved'],2)

if __name__=='__main__':unittest.main()
