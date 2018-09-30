import os
import subprocess

from absl import app, flags
import fsdb
import prep_flags
import shipname

# From rl_loop/fsdb.py
# Must pass one or the other in.
flags.declare_key_flag('bucket_name')
flags.declare_key_flag('base_dir')

def bootstrap(unused_argv):
    bootstrap_name = shipname.generate(0)
    bootstrap_model_path = os.path.join(fsdb.models_dir(), bootstrap_name)
    prep_flags.checked_run([
        'python', 'bootstrap.py',
        '--export_path={}'.format(bootstrap_model_path),
        '--flagfile=rl_loop/distributed_flags'])

if __name__ == '__main__':
    app.run(bootstrap)