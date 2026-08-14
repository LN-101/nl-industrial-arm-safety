from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'),
            glob('models/*.zip')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'PyYAML'],
    scripts=['scripts/control'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
       'ik_control = control.ik_control:main',
        'drl_control = control.drl_control:main',
        'handeye_calibration = control.handeye_calibration:main',
        ],
    },
)
