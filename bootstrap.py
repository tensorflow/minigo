import os

from absl import app, flags
import dual_net
import utils

flags.DEFINE_string('export_path', None,
                    'Where to export the model after training.')

flags.declare_key_flag('work_dir')

FLAGS = flags.FLAGS

def bootstrap(export_path):
    dual_net.bootstrap()
    dual_net.export_model(export_path)

def main(unused_argv):
    utils.ensure_dir_exists(os.path.dirname(FLAGS.export_path))
    bootstrap(FLAGS.export_path)

if __name__ == '__main__':
    flags.mark_flags_as_required(['work_dir', 'export_path'])
    app.run(main)