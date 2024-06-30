from setuptools import find_packages, setup

setup(
    name="shakespeare-generator",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch",
        "fastapi",
        "uvicorn",
    ],
    author="sochoav8a",
    author_email="sochoav8a@email.com",
    description="Un generador de texto al estilo de Shakespeare para practicar tanto redes neuronales como despliegues de modelos en la nube.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sochoav8a/shakespeare-generator",
)