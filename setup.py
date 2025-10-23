from setuptools import find_packages
from skbuild import setup


package_name = 'apriltag_pose_estimation'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=[package_name, f'{package_name}.*'], exclude=['*.testcase']),
    install_requires=[
        'setuptools',
        'numpy<2',
        'opencv-python',
        'scipy'
    ],
    extras_require={
        'generate': [
            'Pillow',
            'fpdf2'
        ]
    },
    python_requires='>=3.10',
    zip_safe=False,
    author='The Mechatronics and Robotics Society at the University of Virginia',
    description='A Python library for pose estimation using AprilTags.'
)
