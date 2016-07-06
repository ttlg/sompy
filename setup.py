
from setuptools import setup, find_packages
version = '0.1.0'

try:
    import pypandoc
    read_md = lambda f: pypandoc.convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(
        name='sompy',
        version=version,
        description="Numpy based SOM Library",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Topic :: Software Development :: Libraries :: Python Modules",
            #"Topic :: Internet :: WWW/HTTP :: Dynamic Content :: CGI Tools/Libraries",
            #"Topic :: Machine Learning",
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
        long_description=read_md('README.md'),
        install_requires=["numpy"]
)
