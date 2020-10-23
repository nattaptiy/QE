import re
import platform
from config.mlqe.const import *
from config.mlqe.datapath import *
from config.mlqe.const import DOMAIN

# argparseで引数を受け取った場合、宣言した定数を上書き
for key, value in args.load_const(DOMAIN).items():
    print("%s:OVERWRITE" % DOMAIN, key, value)
    exec("%s = %s" % (key, value))

if platform.system() == "Darwin":
    DEVICE = "cpu"

print("#" * 100)
print("hyper parameter and const")
regex = re.compile("[A-Z]+")
for key in sorted(locals().keys()):
    if regex.match(key):
        print("%s:" % key, locals()[key])
print("\n" * 3)
