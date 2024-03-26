import setuptools
from importlib.util import module_from_spec, spec_from_file_location

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


spec = spec_from_file_location("constants", "./src/canalyst_candas/version.py")
constants = module_from_spec(spec)
spec.loader.exec_module(constants)

__version__ = constants.__version__


setuptools.setup(
    name="canalyst_candas",
    version=__version__,
    author="Canalyst",
    author_email="support+api@canalyst.com",
    description="The official Canalyst Software Development Kit (SDK) for our public API",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    include_package_data=True,
    install_requires=[
        "boto>=2.49.0",
        "boto3>=1.17.27",
        "botocore>=1.20.27",
        "fred>=3.1",
        "fredapi>=0.4.3",
        "future>=0.18.2",
        "graphviz>=0.16",
        "joblib>=1.0.1",
        "networkx>=2.5.1",
        "numexpr>=2.7.1",
        "numpy>=1.19.2",
        "pandas>=1.2.3",
        "plotly>=4.14.3",
        "pydantic>=1.8.2",
        "pydot>=1.4.2",
        "python_graphql_client>=0.4.3",
        "pyvis>=0.1.9",
        "openpyxl>=3.0.7",
        "requests>=2.24.0",
        "requests_html>=0.10.0",
        "urllib3>=1.25.11",
        "matplotlib>=3.4.3",
        "statsmodels>=0.12.2",
        "yahoo-fin>=0.8.9.1",
        "pyarrow==6.0.1",
    ],
)
