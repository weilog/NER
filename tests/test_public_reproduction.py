"""Public material restoration tests use synthetic fixtures, not experiment data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec=importlib.util.spec_from_file_location('public_restore',Path(__file__).resolve().parents[1]/'scripts/reproduce_public.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class PublicRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
    def tearDown(self):self.temporary.cleanup()
    def fixture(self):
        data=b'{"fixture":true}\n';digest=hashlib.sha256(data).hexdigest()
        snapshot=self.root/'reports/demo_inputs_manifest.json';snapshot.parent.mkdir()
        snapshot.write_text(json.dumps({'files_sha256':{'data/test.jsonl':digest}}),encoding='utf-8')
        archive=self.root/'complete.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('data/test.jsonl',data)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps({'sha256':{'data/test.jsonl':digest}}))
        raw=archive.read_bytes();part=self.root/'reproducibility/one.part';part.parent.mkdir();part.write_bytes(raw)
        return archive,{'archive_name':'restored.zip','archive_bytes':len(raw),'archive_sha256':hashlib.sha256(raw).hexdigest(),
                        'file_count':1,'parts':[{'path':'reproducibility/one.part','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]}
    def test_restore_and_final_hashes(self):
        archive,manifest=self.fixture();result=module.restore(self.root,archive,manifest)
        self.assertEqual(result['final_inputs_verified'],1)
        self.assertTrue((self.root/'data/test.jsonl').exists())
    def test_part_reassembly(self):
        archive,manifest=self.fixture();restored=module.assemble(self.root,manifest)
        self.assertEqual(module.sha256(restored),module.sha256(archive))
    def test_tampered_part_is_rejected(self):
        _,manifest=self.fixture();(self.root/'reproducibility/one.part').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'SHA256/size mismatch'):module.assemble(self.root,manifest)
    def test_existing_required_file_is_not_overwritten(self):
        archive,manifest=self.fixture();target=self.root/'data/test.jsonl';target.parent.mkdir();target.write_bytes(b'personal change')
        with self.assertRaisesRegex(RuntimeError,'preserved without overwrite'):module.restore(self.root,archive,manifest)
        self.assertEqual(target.read_bytes(),b'personal change')
    def test_path_traversal_is_rejected(self):
        for path in ['../escape','/absolute','C:/escape','data\\escape']:
            with self.subTest(path=path),self.assertRaises(RuntimeError):module.safe_path(self.root,path)
    def test_actual_env_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'credential file'):module.safe_path(self.root,'data/.env')

if __name__=='__main__':unittest.main()
