from setuptools import setup, find_packages
from os import path
this_directory = path.abspath(path.dirname(__file__))

long_description = None
with open(path.join(this_directory, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required = None
with open(path.join(this_directory, 'requirements.txt')) as f:
    required = f.read().splitlines()

setup(name='trtpg',
      packages=find_packages(),
      version='1.3.0',
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3',
      ],
      install_requires=required,
      entry_points={'console_scripts': ['trtpg = tpg.tpg:main']},
      platforms="linux",
      package_data={'tpg': ['plugin_templates/*']},
      auth='zeroz',
      author_email='cntse@nvidia.com',
      description='Generate TensorRT plugin in fly',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/NVIDIA-AI-IOT/tensorrt_plugin_generator',
      keywords='tensorrt plugin generator'
)
