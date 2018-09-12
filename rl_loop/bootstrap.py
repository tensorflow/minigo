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
    sp_cmd = [
        'python', 'bootstrap.py',
        '--bootstrap_export_path={}'.format(bootstrap_model_path),
        '--flagfile=rl_loop/distributed_flags']
    completed_process = prep_flags.run(sp_cmd)
    if completed_process.returncode > 0:
        print("Bootstrap failed...")
        print("stdout:\n", completed_process.stdout)
        print("stderr:\n", completed_process.stderr)

if __name__ == '__main__':
    app.run(bootstrap)