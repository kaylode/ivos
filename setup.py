import setuptools

with open('requirements.txt') as f:
    install_reqs = f.read().splitlines()

setuptools.setup(
    name="theseus",
    version='0.0.1',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=install_reqs,
)