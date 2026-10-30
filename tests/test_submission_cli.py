"""Synthetic local-file/direct-download checks; no real network or model APIs."""
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

ENTRY=Path(__file__).resolve().parents[1]/'scripts/reproduce_submission.py'
sys.path.insert(0,str(ENTRY.parent))
spec=importlib.util.spec_from_file_location('single_resource_entry',ENTRY)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class SubmissionCliTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name)/'source';self.root.mkdir()
  self.body=b'synthetic zip bytes'
  self.catalog={'archive_name':'resources.zip','archive_bytes':len(self.body),'archive_sha256':hashlib.sha256(self.body).hexdigest()}
 def test_adjacent_single_zip_is_found(self):
  p=self.root.parent/'resources.zip';p.write_bytes(self.body)
  self.assertEqual(module.locate_resource(self.root,self.catalog),p)
 def test_explicit_directory_containing_zip_is_supported(self):
  folder=self.root.parent/'external';folder.mkdir();p=folder/'resources.zip';p.write_bytes(self.body)
  self.assertEqual(module.locate_resource(self.root,self.catalog,folder),p)
 def test_missing_zip_is_actionable_error(self):
  with self.assertRaisesRegex(RuntimeError,'--materials'):module.locate_resource(self.root,self.catalog)
 def test_download_size_or_hash_mismatch_is_not_installed(self):
  target=self.root/'resources.zip'
  with self.assertRaisesRegex(RuntimeError,'SHA256|size|declared'):
   module.download_resource('https://example.com/resources.zip',target,self.catalog,opener=lambda u,timeout:io.BytesIO(b'bad'))
  self.assertFalse(target.exists())
 def test_non_https_is_rejected_without_opening(self):
  with self.assertRaisesRegex(RuntimeError,'HTTPS'):
   module.download_resource('http://example.com/resources.zip',self.root/'resources.zip',self.catalog)

if __name__=='__main__':unittest.main()
