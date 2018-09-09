import os

from absl import app, flags
import dual_net
import utils


# This should be named export_path but it conflicts with the same
# flag defined in train.py.
# Can be renamex once rl_loop/local_rl_loop stop directly importing
# both bootstrap.py and train.py and instead use a subprocess.
flags.DEFINE_string('bootstrap_export_path', None,
                    'Where to export the model after training.')

flags.declare_key_flag('model_dir')

FLAGS = flags.FLAGS

def bootstrap(export_path):
    dual_net.bootstrap()
    dual_net.export_model(export_path)

def main(unused_argv):
    utils.ensure_dir_exists(os.path.dirname(FLAGS.bootstrap_export_path))
    bootstrap(FLAGS.bootstrap_export_path)

if __name__ == '__main__':
    flags.mark_flags_as_required(['model_dir', 'bootstrap_export_path'])
    app.run(main)