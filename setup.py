from setuptools import setup, find_packages

setup(
    name='demoscode',
    version='0.1',
    packages=find_packages(where='web/services'),
    package_dir={'': 'web/services'},
    include_package_data=True,
    install_requires=[
        # Aquí puedes listar las dependencias, si las hay
    ],
    entry_points={
        'console_scripts': [
            # Si tienes scripts ejecutables, los defines aquí
        ],
    },
)

