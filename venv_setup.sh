rm -rf ./venv
/opt/homebrew/Cellar/python@3.11/3.11.9_1/bin/python3.11 -m venv ./venv
export PATH="$(pwd)/venv/bin":$PATH
echo "$PATH"
python -m pip install --upgrade pip

