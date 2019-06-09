#!/usr/bin/env python3

# Copyright (c) Alexander Sax, Bradley Emi, Jeffrey Zhang, Amir R. Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os.path
import sys

from distutils.core import setup


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "visualpriors"))

with open("README.md", encoding="utf8") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

with open("requirements.txt") as f:
    reqs = f.read()

DISTNAME = "visualpriors"
LONG_DESCRIPTION = readme
REQUIREMENTS = reqs.strip().split("\n")

if __name__ == "__main__":
    setup(
      name =         DISTNAME,
      packages =     [DISTNAME],
      version =      '0.1',
      license=       'MIT',
      description =  'The official implementation of visual priors from the paper Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies. Arxiv preprint 2018. ',   # Give a short description about your library
      long_description=LONG_DESCRIPTION,
      author =       'Alexander Sax, Bradley Emi, Jeffrey Zhang, Amir R. Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik', 
      url =          'https://github.com/alexsax/midlevel-reps',
      download_url = 'https://github.com/alexsax/visual-prior/archive/v0.1-alpha.1.tar.gz',
      keywords =     ['computer vision',
                      'robotics',
                      'perception',
                      'midlevel',
                      'mid-level',
                      'reinforcement learning',
                      'machine learning', 
                      'representation learning'],
      install_requires=REQUIREMENTS,
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License', 
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
      ],
    )