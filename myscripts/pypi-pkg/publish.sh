# twine upload --repository testpypi  src/rai_extensions/rai_perception/dist/*.tar.gz

# to test
# pip install --index-url https://test.pypi.org/simple/ rai-perception
# https://test.pypi.org/project/rai-perception/0.1.1/

pytest tests/pypi_packages/test_build_publish.py -m "" -s -v