from setuptools import setup, find_packages
## Install package api_visual_dnn and model_layers_visualizer
setup(
    name='demoscode',
    version='0.1',
    packages=find_packages(where='web/services'),
    package_dir={'': 'web/services'},
    include_package_data=True,
    install_requires=[
        'api_visual_dnn',
        'model_layers_visualizer',
        'activation_maps_cnn',
    ],
    entry_points={
        'console_scripts': [
            # Si tienes scripts ejecutables, los defines aquí
        ],
    },
)

