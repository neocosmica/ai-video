import urllib.request, sys, os
def check(name, url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        r = urllib.request.urlopen(req, timeout=20)
        print(f"OK   {name:22s} {r.status}")
    except Exception as e:
        print(f"FAIL {name:22s} {type(e).__name__} {str(e)[:50]}")

print("=== python ===", sys.version.replace("\n"," "))
print("=== network ===")
check("pytorch-cu128",  "https://download.pytorch.org/whl/cu128/torch/")
check("pytorch-cu128-idx","https://download.pytorch.org/whl/cu128/")
check("hf-mirror",       "https://hf-mirror.com/api/models/Qwen/Qwen3.5-4B")
check("tuna-pypi",       "https://pypi.tuna.tsinghua.edu.cn/simple/transformers/")
check("pypi.org",        "https://pypi.org/simple/")
print("=== disk ===")
import shutil
t,u,f = shutil.disk_usage("C:/")
print(f"C: total={t/2**30:.1f}GB used={u/2**30:.1f}GB free={f/2**30:.1f}GB")
