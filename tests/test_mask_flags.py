# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mask_flags
import test_utils


class TestMaskFlags(test_utils.MiniGoUnitTest):
    def test_pyflags_extraction(self):
        self.assertEqual(mask_flags.parse_helpfull_output('''
            --some_flag: Flag description
             lolnoflag, maybe some help text wrapping or something
            --[no]bool_flag: Flag description
            '''), {'--some_flag', '--bool_flag', '--nobool_flag'})

    def test_ccflags_extraction(self):
        self.assertEqual(mask_flags.parse_helpfull_output('''
          Header stuff that should be ignored
             -some_flag (Flag description)
             lolnoflag, maybe some help text wrapping or something
             -[no]bool_flag (Flag description)
          ''', regex=mask_flags.FLAG_HELP_RE_CC),
                         {'--some_flag', '--bool_flag', '--nobool_flag'})
