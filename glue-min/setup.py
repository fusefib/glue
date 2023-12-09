from setuptools import setup

REQUIRED = [
    'Pillow>=9.1.0', # Works with pillow 10.1.0
    'Jinja2', # Works with jinja2 3.1.2, MarkupSafe 2.1.3
]

setup(
    name='glue-min',
    version='0.13.1',
    url='http://github.com/jorgebastida/glue',
    license='BSD',
    author='Jorge Bastida (original author)',
    author_email='me@jorgebastida.com',
    description='Lightweight version of Glue for generating sprites with CSS.',
    long_description=('glue-min.py is a simple command line tool that generates '
                      'sprites (images + metadata) from source images like PNG, '
                      'JPEG, or GIF. This streamlined Glue variant with '
                      'minimal features is designed for straightforward '
                      'integration into web projects. Variant not endorsed '
                      'by the original author. '),
    keywords='glue sprites css',
    scripts=['glue-min.py'],
    install_requires=REQUIRED,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Utilities'
    ],
)
