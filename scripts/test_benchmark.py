import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class BenchmarkTest(unittest.TestCase):
    def test_paired_runs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data'; models=data/'dashformer_model_parameters'; models.mkdir(parents=True)
            (data/'dashformer_tokenizer.json').write_text('{}')
            (models/'fixture.txt').write_text('fixture')
            examples=root/'input.list'; examples.write_text('a c,0\n')
            binary=root/'fake-inference'
            binary.write_text('#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\nif "--check-inputs" not in sys.argv:\n out=Path(sys.argv[sys.argv.index("--output")+1]); (out/"output.txt").write_text("0.5\\n")\n')
            binary.chmod(0o755)
            out=root/'results'
            command=[sys.executable,str(Path(__file__).with_name('benchmark.py')),'--binary',str(binary),'--data-dir',str(data),'--examples',str(examples),'--output-dir',str(out),'--repeats','2']
            subprocess.run(command,check=True,capture_output=True)
            raw=(out/'manifest.json').read_bytes(); manifest=json.loads(raw)
            self.assertEqual([r['mode'] for r in manifest['runs']],['baseline','optimized','optimized','baseline'])
            self.assertTrue(all(r['exit_code']==0 and 'output_sha256' in r for r in manifest['runs']))
            self.assertEqual(examples.read_text(),'a c,0\n')
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual((out/'manifest.json').read_bytes(),raw)


if __name__=='__main__':unittest.main()
