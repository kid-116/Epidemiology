import pandas as pd
import datetime as dt
import sys
import os
import glob    
import importlib.util

sys.path.append('../')

from src.utils import SIR, run

df_path = 'src/data/generation/contact_network/'
metadata_path = "src/data/generation/metadata/"
def get_path(path):
    retval = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            retval.append(os.path.join((path), file))
            req_path = max(retval)
            file = pd.read_csv(req_path)
    return file

get_path(df_path)
get_path(metadata_path)

     

def test_model():
    model = SIR(df=get_path(df_path), metadata=get_path(metadata_path))

    dir_path = os.path.dirname('src/vaccination_strategy/*')
    files_in_dir = [
        f[:-3] for f in os.listdir(dir_path)
        if f.endswith('.py') and f != '__init__.py'
    ]

    for f in files_in_dir:
        spec = importlib.util.spec_from_file_location('module.name', f'src/vaccination_strategy/{f}.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        vaccination_strategy = mod.vaccination_strategy
        vaccinated = vaccination_strategy(model)
        
        result = run(model, vaccinated, 5)
        assert 'metrics' in result.keys()
        assert 'total_deaths' in result['metrics'].keys()
        assert result['metrics']['total_deaths'] >= 0
        assert 'peak_infections' in result['metrics'].keys()
        assert result['metrics']['total_deaths'] >= 0

test_model()