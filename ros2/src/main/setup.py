from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'main'

def is_conda_env():
    return 'CONDA_PREFIX' in os.environ

if is_conda_env():
    install_requires = ['setuptools']
else:
    install_requires = ['setuptools']

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=install_requires,  # 使用动态依赖
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='Demo package for arm control',
    license='Apache License 2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'arm_state=main.arm_state:main',
            'estop_aggregator=main.estop_aggregator:main',

        ],
    },
)
