"""Submission ZIP guard tests use synthetic files, not real keys or data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

ENTRY=Path(__file__).resolve().parents[1]/'scripts/build_submission_zip.py'
class SubmissionZipTests(unittest.TestCase):
 def setUp(self):
  self.assertTrue(ENTRY.is_file(),'Missing source-only ZIP builder')
  spec=importlib.util.spec_from_file_location('submission_zip',ENTRY);self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'source';self.root.mkdir()
 def put(self,relative,body='synthetic'):
  p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body,encoding='utf-8');return p
 def test_collect_excludes_all_heavy_and_secret_files(self):
  self.put('README.md');self.put('src/a.py');self.put('.env.example','GLM_API_KEY=\n')
  for name in ['.env','data/train.jsonl','runs/model.pkl','.venv/pyvenv.cfg','reproducibility/materials/x.part','reproducibility/data.zip','checkpoints/x.pt','video.mp4']:
   self.put(name)
  names={p.relative_to(self.root).as_posix() for p in self.m.collect(self.root)}
  self.assertEqual(names,{'README.md','src/a.py','.env.example'})
 def test_actual_key_bytes_block_archive(self):
  secret=b'synthetic_private_token_value';self.put('README.md',secret.decode())
  target=Path(self.temp.name)/'output.zip'
  with self.assertRaisesRegex(RuntimeError,'Credential'):
   self.m.build(self.root,target,[secret])
  self.assertFalse(target.exists())
 def test_package_crc_and_member_hash_inventory(self):
  self.put('README.md','hello');self.put('src/a.py','print(1)')
  target=Path(self.temp.name)/'output.zip';result=self.m.build(self.root,target,[])
  self.assertEqual(result['status'],'PASS')
  with zipfile.ZipFile(target) as z:
   self.assertIsNone(z.testzip());self.assertEqual(set(z.namelist()),{'README.md','src/a.py','SOURCE_PACKAGE_MANIFEST.json'})
 def test_empty_key_template_is_required(self):
  self.put('.env.example','GLM_API_KEY=not_empty\n')
  with self.assertRaisesRegex(RuntimeError,'template'):
   self.m.build(self.root,Path(self.temp.name)/'out.zip',[])
if __name__=='__main__':unittest.main()
