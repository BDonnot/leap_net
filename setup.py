# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import sys
import setuptools
from setuptools import setup

pkgs = {
    "required": [
        "keras>=3",
        "numpy",
        "scipy",
        "packaging"
    ],
    "extras": {
        "recommended": [
            "grid2op",
            "pandas",
            "tqdm",
            'scikit-learn',
            "tensorflow"
        ]
    }
}

if sys.version_info.major == 3 and sys.version_info.minor == 8:
    # no keras v3 in python 3.8
    pkgs["required"] = [el for el in pkgs["required"] if not "keras" in el]
    pkgs["required"].append("tensorflow")
    
    
pkgs["extras"]["test"] = [el for el in pkgs["extras"]["recommended"] if not "tensorflow" in el]
# from here https://keras.io/getting_started/
# If you install TensorFlow, critically, you should reinstall Keras 3 afterwards. 
# This is a temporary step while TensorFlow is pinned to Keras 2, 
# and will no longer be necessary after TensorFlow 2.16. The cause is that tensorflow==2.15 
# will overwrite your Keras installation with keras==2.15.
# This is why in the tests I just skip tensorflow for now. Will be fixed later

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(name='leap_net',
      version='0.1.1',
      description='An implementation in keras 3.0 (and tensorflow keras) of the LeapNet model',
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='LEAP-Net guided-dropout dropout resnet',
      author='Benjamin DONNOT',
      author_email='benjamin.donnot@rte-france.com',
      url="https://github.com/bdonnot/leap_net",
      python_requires='>=3.8',
      license='Mozilla Public License 2.0 (MPL 2.0)',
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=pkgs["required"],
      extras_require=pkgs["extras"],
      zip_safe=False,
      entry_points={
          'console_scripts': []
     }
)
