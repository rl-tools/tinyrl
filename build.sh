rm -rf ../tinyrl/dist
pip install --upgrade build twine
python3 -m build
python3 -m twine upload dist/* --password $PYPI_TOKEN