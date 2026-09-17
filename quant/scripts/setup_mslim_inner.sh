set -e
H=/home/user; V=$H/projects/dsv41/env/mslim-venv
python3 -V; command -v python3
SYS=$(python3 -c "import site as s; print(s.getsitepackages()[0])")
echo "system site-packages: $SYS"
if [ ! -x $V/bin/python ]; then rm -rf $V; python3 -m venv $V; fi
SITE=$($V/bin/python -c "import site as s; print(s.getsitepackages()[0])")
echo "$SYS" > "$SITE/system.pth"
printf '[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple/\ntrusted-host = mirrors.aliyun.com\n' > $V/pip.conf
$V/bin/pip install -q -U pip setuptools wheel 2>&1 | tail -1
$V/bin/pip install -q -e $H/projects/dsv41/src/msmodelslim 2>&1 | tail -2
echo "=== verify ==="
$V/bin/python -c "import msmodelslim, torch, torch_npu, transformers; print('OK | torch', torch.__version__, '| transformers', transformers.__version__)"
$V/bin/msmodelslim --help 2>&1 | head -6
