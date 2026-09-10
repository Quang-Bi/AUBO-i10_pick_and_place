from setuptools import find_packages, setup

package_name = 'gymnasium_auboi10'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='quang',
    maintainer_email='quang@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test_env = gymnasium_auboi10.test_env:main',
            'aubo_env = gymnasium_auboi10.aubo_env:main',
            'train_sac = gymnasium_auboi10.train_sac:main',
            
        ],
    },
)
