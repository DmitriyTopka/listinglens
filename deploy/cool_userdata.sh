#!/bin/bash
# EC2 user-data for the COOL vs stock OpenCV 5 benchmark on Graviton4 (c8g). Self-terminates.
shutdown -h +150
exec > /var/log/ll.log 2>&1
set -x
say() { echo "$@" | tee /dev/console /dev/ttyS0 >/dev/null 2>&1; echo "$@"; }
mkdir -p /root/ll && cd /root/ll
# (source files are inlined here as heredocs by deploy/make_cool_userdata.py)
__CODE__
say "LLSTAGE code ok"
# COOL must be used through its activate script (it sets PYTHONPATH / LD_LIBRARY_PATH), see the AMI usage notes
say "LLDIAG $(ls /opt/cool /opt/cool/venvs 2>&1 | tr '\n' ' ')"
say "LLDIAG $(grep -E 'PYTHONPATH|LD_LIBRARY_PATH' /opt/cool/venvs/python_3.12/bin/activate 2>&1 | tr '\n' ' ' | cut -c1-300)"
cat > /root/coolpy <<'SH'
#!/bin/bash
source /opt/cool/venvs/python_3.12/bin/activate
export LD_LIBRARY_PATH="/opt/cool/cpp_sdk/lib:${LD_LIBRARY_PATH:-}"
python -c 'import cv2' 2>/dev/null || export PYTHONPATH="/opt/cool/python_3.12/site-packages:${PYTHONPATH:-}"
exec python "$@"
SH
chmod +x /root/coolpy
COOLPY=/root/coolpy
$COOLPY -c 'import numpy' 2>/dev/null || /opt/cool/venvs/python_3.12/bin/python -m pip install -q numpy==2.2.6
say "LLCOOL $($COOLPY -c 'import cv2,numpy;print(cv2.__version__, cv2.__file__, numpy.__version__)' 2>&1 | tail -1)"
apt-get install -y -q python3.12-venv >/dev/null 2>&1 || true
python3.12 -m venv /opt/stock && /opt/stock/bin/pip install -q opencv-python-headless==5.0.0.93 numpy==2.2.6
say "LLSTOCK $(/opt/stock/bin/python -c 'import cv2,numpy;print(cv2.__version__, cv2.__file__, numpy.__version__)' 2>&1 | tail -1)"
mkdir -p imgs
while read id p; do curl -s -f -o imgs/$id.jpg https://amazon-berkeley-objects.s3.amazonaws.com/images/original/$p; done < bench/abo_bench_images.txt
say "LLSTAGE images $(ls imgs | wc -l)"
say "LLSTAGE cpu $(lscpu | grep -E 'Model name|^CPU\(s\)' | tr -s ' ' | tr '\n' ' ')"
for r in 1 2 3; do
  for env in cool stock; do
    PY=$COOLPY; [ $env = stock ] && PY=/opt/stock/bin/python
    $PY bench/bench_cool.py --label $env-r$r --images imgs --n 20 2>&1 | grep LLRESULT | while read l; do say "$l"; done
  done
done
say "LLSTAGE done"
# keep the console readable for 15 minutes so the results can be collected, then terminate
sleep 900
shutdown -h now
