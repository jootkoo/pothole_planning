import json,random,tempfile,unittest
from pathlib import Path
from .generator import build_one,generate
from .scenario_validator import validate
class GeneratorTests(unittest.TestCase):
    def setUp(self):self.config=json.loads(Path(__file__).with_name('config.json').read_text())
    def test_geometry(self):
        for i in range(40):
            data,_=build_one(i,42+i*104729,self.config)
            self.assertEqual(validate(data),[])
    def test_repeatable(self):
        a,_=build_one(12,333,self.config);b,_=build_one(12,333,self.config)
        self.assertEqual(a,b)
    def test_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(generate(3,42,tmp,self.config),12)
            self.assertEqual(len(list((Path(tmp)/'experiments').glob('*.json'))),12)
            self.assertTrue((Path(tmp)/'features.csv').exists())
if __name__=='__main__':unittest.main()
