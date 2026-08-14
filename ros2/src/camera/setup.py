from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 安装模型文件
        (os.path.join('share', package_name, 'models'),
         glob('models/*.pt') + glob('models/*.onnx')),
        # 安装配置文件
        (os.path.join('share', package_name),
         [path for path in ['safety_rules.json', 'arm_rules.json'] if os.path.exists(path)]),
        # 安装orbbec库文件到lib目录
        (os.path.join('lib', package_name),
         glob('camera/lib/*.so')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='Camera node with YOLO detection',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'color_stop = camera.color_stop:main',
            'min_dis = camera.min_dis:main',
            'k230 = camera.k230:main'

        ],
    },
)
