from setuptools import setup, find_packages
version = '0.1.1'

setup(
        name='sompy',
        version=version,
        description="Numpy based SOM Library",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Education",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering",
            "License :: OSI Approved :: MIT License",
            ],
        keywords='som, self organizing map, machine learning',
        author='Yota',
        author_email='dev.ttlg@gmail.com',
        url='https://github.com/ttlg/sompy',
        license='MIT',
        packages=find_packages(exclude=['examples', 'tests']),
        include_package_data=True,
        zip_safe=True,
        install_requires=["numpy"]
)
