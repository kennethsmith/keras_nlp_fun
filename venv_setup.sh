rm -rf ./venv
/usr/local/bin/python3.11 -m venv ./venv
export PATH="$(pwd)/venv/bin":$PATH
echo "$PATH"
python -m pip install --upgrade pip
