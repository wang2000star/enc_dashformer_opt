from pathlib import Path
import tempfile
import unittest
from compare_outputs import compare


class CompareOutputsTest(unittest.TestCase):
    def test_differences_and_agreement(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'a',Path(temp)/'b'
            a.write_text('1 3\n4 2\n');b.write_text('2 2\n4 2\n')
            r=compare(a,b)
            self.assertEqual(r['rows'],2)
            self.assertEqual(r['max_absolute_difference'],1)
            self.assertAlmostEqual(r['root_mean_square_difference'],(0.5)**0.5)
            self.assertEqual(r['top1_agreement'],0.5)

    def test_invalid_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'a',Path(temp)/'b';a.write_text('1 3\n')
            for invalid in ['','NaN 1\n','1\n','1 3\n1\n','Inf 1\n']:
                b.write_text(invalid)
                with self.assertRaises(ValueError):compare(a,b)


if __name__=='__main__':unittest.main()
