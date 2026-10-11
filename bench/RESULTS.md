# COOL vs stock OpenCV 5 on AWS Graviton4 (final run, 11 Oct 2026)

- Instance: EC2 c8g.large (2 vCPU AWS Graviton4, Neoverse-V2, 2.8 GHz), us-east-1, instance i-09c2d9b5098b7428e,
  AMI from AWS Marketplace "Cloud Optimized OpenCV For AWS Graviton4" v3.1
  (`resolve:ssm:/aws/service/marketplace/prod-k6u24vxijyzvg/cloud-optimized-opencv-3.1`).
- COOL: `/opt/cool/venvs/python_3.12` (activated), `cv2.__version__ = 5.1.0-dev`,
  `cv2.__file__ = /opt/cool/python_3.12/site-packages/cv2/python-3.12/cv2.cpython-312-aarch64-linux-gnu.so`, numpy 2.5.1.
- Stock: `/opt/stock` venv, `opencv-python-headless==5.0.0.93` (KleidiCV enabled in the wheel too), numpy 2.2.6.
- Code: commit 34386af (`bench/bench_cool.py`), 20 catalog photos x 13 variants = 260 images per agent run,
  rounds alternate cool/stock, 3 rounds each. Medians over rounds; speedup = stock / COOL.
- Every one of the 6 runs produced the same decision hash `f6c1d93bb5b62db5` (identical decisions).
- Raw JSON lines: CloudShell `~/final.jsonl` (collected from the serial console by `bench/parse_console.py`).

rounds: cool 3, stock 3; cool 5.1.0-dev | stock 5.0.0

| operation | COOL ms | stock ms | speedup |
|---|---:|---:|---:|
| resize_area_1600_to_1024 | 6.002 | 5.984 | 1.00x |
| resize_lanczos_up_2x | 24.560 | 24.233 | 0.99x |
| cvtColor_BGR2LAB | 3.829 | 3.839 | 1.00x |
| cvtColor_BGR2GRAY | 0.429 | 0.415 | 0.97x |
| GaussianBlur_s1.6 | 10.547 | 9.942 | 0.94x |
| blur_box_1x9 | 2.744 | 7.862 | 2.87x |
| Laplacian_64F | 2.610 | 2.596 | 0.99x |
| filter2D_3x3 | 3.334 | 3.318 | 1.00x |
| Canny | 2.296 | 2.075 | 0.90x |
| threshold_otsu | 0.856 | 1.658 | 1.94x |
| adaptiveThreshold_gauss | 11.586 | 8.973 | 0.77x |
| morphologyEx_close | 1.962 | 2.002 | 1.02x |
| findContours | 0.158 | 0.174 | 1.10x |
| warpAffine | 5.989 | 5.917 | 0.99x |
| addWeighted_unsharp | 2.970 | 2.815 | 0.95x |
| dct_32x32 | 0.012 | 0.007 | 0.58x |
| createCLAHE_apply | 4.755 | 6.663 | 1.40x |
| imencode_jpeg | 5.975 | 6.109 | 1.02x |
| grabCut_1024_2it | 720.757 | 761.064 | 1.06x |
| fastNlMeansDenoisingColored_1024 | 764.282 | 778.337 | 1.02x |
| agent p50_ms | 169.8 | 168.0 | 0.989x |
| agent p95_ms | 1226.2 | 1328.9 | 1.084x |
| agent mean_ms | 344.5 | 356.1 | 1.034x |
| agent throughput_img_per_s | 2.903 | 2.808 | 1.034x |

An earlier run on commit e6c3610 (GrabCut at 1024 px, 4 iterations) gave the same picture: agent mean
570.4 vs 586.3 ms (1.028x), box blur 2.91x, Otsu 1.92x, CLAHE 1.40x, adaptiveThreshold 0.77x; decision hash
`b7212930db064904` identical across builds.

## Per-stage profile (run 2, 11 Oct 2026, instance i-0092216293b482da2, code commit ea1c379 + ast-minified)

Same setup (c8g.large, COOL 3.1 vs stock 5.0.0.93, 3 + 3 alternating rounds, 260 photos each). Seconds spent
per stage over the 260-photo batch (median over rounds). Decision hash `547e5491eb6ccf36`, identical in all six runs.

| stage | COOL s | stock s | speedup |
|---|---:|---:|---:|
| perceive (all measurements, also re-run after every action) | 30.12 | 29.97 | 0.99x |
| tool: denoise (fastNlMeansDenoisingColored) | 20.29 | 22.27 | 1.10x |
| tool: whiten_background (GrabCut) | 17.07 | 17.21 | 1.01x |
| tool: exposure | 2.55 | 2.60 | 1.02x |
| tool: recrop | 1.17 | 1.16 | 0.99x |
| tool: white_balance / sharpen | 0.39 | 0.38 | 0.97x |
| **whole agent, mean ms per photo** | **299.4** | **307.0** | **1.025x** |
| whole agent, throughput img/s | 3.339 | 3.257 | 1.025x |

Micro-benchmarks repeated within 1 %: box blur 2.87x, Otsu 1.92x, CLAHE 1.40x, grabCut 1.06x, NLM 1.02x.

**Why perception does not speed up.** A cProfile of `perceive()` on 40 photos (Mac, 2 threads) shows that
about 80 % of its time is NumPy and Python (LAB distance with `np.linalg.norm`, percentiles, dtype
conversions, boolean masks) and only about 17 % is OpenCV calls (cvtColor, resize, morphology, blur, Sobel,
Canny). COOL accelerates OpenCV, so it cannot touch most of this stage. Moving those NumPy steps into
OpenCV primitives is the change that would let COOL's 2-3x kernels show up end to end.
