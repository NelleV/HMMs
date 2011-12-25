from setuptools import setup

version = '0.0.0'
long_description = '\n\n'.join([open('README.rst').read(),
                                open('CHANGES.rst').read(),
                                open('TODO.rst').read()])

setup(name='hmms',
      version=version,
      description="HMMs",
      long_description=long_description,
      classifiers=[],
      keywords='HMM',
      author='Nelle Varoquaux',
      author_email='nelle.varoquaux@gmail.com',
      namespace_packages=[],
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'setuptools',
          # -*- Extra requirements: -*-
      ],
      entry_points={},
      )
