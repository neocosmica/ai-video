import os, sys, io, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
t,u,f = shutil.disk_usage("C:/")
print("C: total=%.1fGB used=%.1fGB free=%.1fGB" % (t/2**30,u/2**30,f/2**30))
hub = r"C:/Users/annas/.cache/huggingface/hub"
if os.path.exists(hub):
    sz=0
    for root,_,files in os.walk(hub):
        for fn in files:
            try: sz+=os.path.getsize(os.path.join(root,fn))
            except OSError: pass
    print("HF cache: %.1fGB" % (sz/2**30))
    for d in sorted(os.listdir(hub)):
        p=os.path.join(hub,d); s=0
        if os.path.isdir(p):
            for r_,_,fs in os.walk(p):
                for x in fs:
                    try: s+=os.path.getsize(os.path.join(r_,x))
                    except OSError: pass
            if s>100*1024**2: print("  %-60s %.2fGB" % (d, s/2**30))
