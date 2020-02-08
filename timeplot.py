#!/usr/bin/env python3

from sys import argv
from math import log
import numpy as np
import glob
from datetime import datetime

def parse_file(fname):
    f=open(fname, "r")
    total = 0
    total_deaths = 0
    for line in f.readlines():
        if line[0:9] == "# update:":
            timestamp = line.strip()[10:]
        if line[0] != "#" and not line.find("TOTAL")>=0:
            sline = line.split("|")
            total+=int(sline[1].replace(",",""))
            total_deaths+=int(sline[2].replace(",",""))
    return (timestamp, total, total_deaths)
# update: 2020-01-24 14:55:00 CST


filenames = glob.glob(argv[1])
values = {}
for fname in filenames:
    try:
        (timestamp, total, total_deaths) = parse_file(fname)
        ts = datetime(int(timestamp[0:4]), int(timestamp[5:7]), int(timestamp[8:10]),
                         int(timestamp[11:13]), int(timestamp[14:16]), int(timestamp[17:19]) ) 
        values[int(ts.timestamp())] = (total, total_deaths, log(total), log(totaal_deaths))
    except Exception:
        continue


for i in sorted(values.keys()):
    v=values[i]
    print(f"{i}\t" + "\t".join(v))
#
# 1-st approa
#
#  


