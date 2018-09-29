# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================


import os
from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

class CustomInstallCommand(install):
    '''a customized install class in pip module'''
    user_options = install.user_options + [
        ('sdk-only', None, '<add --sdk-only if you want to only install nni sdk')
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.sdk_only = None

    def install_requires(self):
        self.install_requires_list = [
            'astor',
            'hyperopt',
            'json_tricks',
            'numpy',
            'psutil',
            'pyyaml',
            'requests',
            'scipy',
            'schema',
            'pyhdfs'
        ]
        for pkg in self.install_requires_list:
            subprocess.run(['python3', '-m', 'pip', 'install', pkg], check=True)

    def run(self):
        super().run()
        if self.sdk_only is None:
            subprocess.run(['make', 'pip-install'], check=True)
        self.install_requires()

setup(
    name = 'NNI',
    version = '0.2.0',
    author = 'Microsoft NNI Team',
    author_email = 'nni@microsoft.com',
    description = 'Neural Network Intelligence project',
    long_description = read('README.md'),
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',

    packages = find_packages('src/sdk/pynni', exclude=['tests']) + find_packages('tools'),
    package_dir = {
        'nni_annotation': 'tools/nni_annotation',
        'nni': 'src/sdk/pynni/nni',
        'nnicmd': 'tools/nnicmd',
        'trial_tool':'tools/trial_tool'
    },
    package_data = {'nni': ['**/requirements.txt']},
    python_requires = '>=3.5',
    

    cmdclass={
        'install': CustomInstallCommand
    }
)
