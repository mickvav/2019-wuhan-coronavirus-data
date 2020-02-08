#!/usr/bin/env python3

from sys import argv
from math import log, exp
from collections import defaultdict
import numpy as np
import glob
import matplotlib.pyplot as plt
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
        if total_deaths > 2:
            values[int(ts.timestamp())] = (total, total_deaths, log(total), log(total_deaths))
    except Exception as e:
        print(e)
        continue


for i in sorted(values.keys()):
    v=values[i]
    print(f"{i}\t" + "\t".join(str(k) for k in v))
#
# 1-st approach:
#
#  ^log(deaths)
#  |     * *
#  |    * 
#  |  **
#  | *
#  +-------->(time)
#  y=k*t+y0
#    ?   ?
# 
#  linalg.lstsq(a,b) expects
#  a*x=b
#    ?   notation
#   
#  So,
#  a = [[t[i], 1]]
#  b = y
# 
#  k = x[0]
#  y0 = x[1]
#  Note - offsetting epoch, dividing time by 24*3600

ts = sorted(values.keys())
t0=ts[0]
dt=24*3600  

def plot_linear(vls, column=3, label="fit"):
    ts = sorted(vls.keys())
    t0=ts[0]
    A = np.vstack([[(t-t0)/dt for t in ts], np.ones(len(ts))]).T
    B = np.array([vls[t][column] for t in ts])
    X, residuals, rank, s = np.linalg.lstsq(A,B)

    if "--show" in argv:
        _ = plt.plot([(t-t0)/dt for t in ts], [vls[t][column] for t in ts], 'o', label="Data", markersize=3)
        _ = plt.plot([(t-t0)/dt for t in ts], [X[0]*(t-t0)/dt+X[1] for t in ts], 'r', label=label)
        _ = plt.legend()
        plt.show()
    return X, t0, ts[-1]

plot_linear(values)
#
# 2-nd approach:
#
#  ^log(deaths)
#  |     * *
#  |    * 
#  |  **
#  | *
#  +-------->(time)
#  y=k1*t**2 + k0*t+y0
#    ?         ?   ?
# 
#  linalg.lstsq(a,b) expects
#  a*x=b
#    ?   notation
#   
#  So,
#  a = [[t[i]**2, t[i], 1]]
#  b = y
# 
#  k1 = x[0]
#  k0 = x[1]
#  y0 = x[2]

ts = sorted(values.keys())
t0=ts[0]
A = np.vstack([
    [((t-t0)/dt)**2 for t in ts], 
    [(t-t0)/dt for t in ts], 
    np.ones(len(ts))
]).T
B = np.array([values[t][3] for t in ts])
X, residuals, rank, s = np.linalg.lstsq(A,B)

if "--show" in argv:
    _ = plt.plot([(t-t0)/dt for t in ts], [values[t][3] for t in ts], 'o', label="Data", markersize=3)
    _ = plt.plot([
        (t-t0)/dt for t in ts], 
        [ X[0]*((t-t0)/dt)**2+X[1]*((t-t0)/dt)+X[2] for t in ts], 'r', label="fit")
    _ = plt.legend()
    plt.show()

#
# Let's try to predict via this model...
#

if "--show" in argv:
    _ = plt.plot(range(100),
        [ X[0]*t**2+X[1]*t+X[2] for t in range(100)], 'r', label="prognosis")
    plt.show()
#
# Expected strangeness - we don't think people will rise from dead.
# Conclusion - yes, growth rate declines.
#

#
# Let's look into differential number of deaths [and of logarithms!]
#
def find_2days_ago(v, t):
    tm = t-48*3600
    if tm in v:
        return (tm, v[tm])
    else:
        while tm>=t0:
            tm=tm-1
            if tm in v:
                return (tm, v[tm])
        return (None, None)

values_diff = defaultdict(list)
for t in ts:
    t1,vi = find_2days_ago(values, t)
    if not(t1 is None):
        for v,vp in zip(values[t], vi):
            diff = dt*(v-vp)/(t-t1) # Units - value per day, not per second! 
            values_diff[t].append(diff)

plot_linear(values_diff,0,"diff(total)")
plot_linear(values_diff,1,"diff(total_deaths)")
plot_linear(values_diff,2,"diff(log(total)")
plot_linear(values_diff,3,"diff(log(total_deaths)")

#
# Ok, looks like the most predictable are diff(log(total)) and diff(log(total_deaths))
# And to have some accuracy we need to filter out first ~10 days
#
plot_linear({t:v for t,v in values_diff.items() if t-t0>10*3600*24},2,"diff(log(total)) last")
X, t01, t_max1 = plot_linear({t:v for t,v in values_diff.items() if t-t0>10*3600*24},3,"diff(log(total_deaths)) last")
#
# Let's try to extrapolate this
#

if "--show" in argv:
    _ = plt.plot(range(100),
        [ X[0]*t+X[1] for t in range(100)], 'r', label="prognosis(diff(log(total)deaths))")
    plt.show()
#
# Ok. This means that there is a decline and we expect to have 0 new deaths at:
#
T_stop = -X[1]/X[0] 
print("Max-precision prediction:")
print(f"T_stop={T_stop} [days]")
print("timestamp = ",str(t01+3600*24*T_stop))
d = datetime.fromtimestamp(t01+3600*24*T_stop)
print("Date = ", d.ctime())
# We can try to estimate total death toll.
#                             T_stop
#                            /
# delta(log(total_deaths)) = | prognosis(diff(log(total_deaths)) dT = prognosis(T_last)*(T_stop-T_last)/2
#                            /
#                           T_last
T_last_sec = max(values_diff.keys())
T_last_days = (T_last_sec - t01)/(24*3600)
prognosis_T_last = X[0] * T_last_days + X[1]
delta_log_total_deaths = prognosis_T_last * (T_stop-T_last_days)/2.0
log_total_deaths_prognosed = values[T_last_sec][3] + delta_log_total_deaths
total_deaths_prognosed = exp(log_total_deaths_prognosed)
print(f"total_deaths_prognosed = {total_deaths_prognosed}")

# 
# Let's check our prediction quality. Filtering out last 2 days and predicting again
#
print("2-days-removed prediction:")
tmax = max(values_diff.keys())
X, t02, t_max2 = plot_linear({t:v for t,v in values_diff.items() if t-t0>10*3600*24 and tmax-t > 2*24*3600},3,"diff(log(total_deaths)) last but 2 days")

T_stop = -X[1]/X[0] 
print(f"T_stop={T_stop}")
print("timestamp = ",str(t02+3600*24*T_stop))
d = datetime.fromtimestamp(t02+3600*24*T_stop)
print("Date = ", d.ctime())

T_last_sec = t_max2
T_last_days = (T_last_sec - t02)/(24*3600)
prognosis_T_last = X[0] * T_last_days + X[1]
delta_log_total_deaths = prognosis_T_last * (T_stop-T_last_days)/2.0
log_total_deaths_prognosed = values[T_last_sec][3] + delta_log_total_deaths
total_deaths_prognosed = exp(log_total_deaths_prognosed)
print(f"total_deaths_prognosed = {total_deaths_prognosed}")


print("4-days-removed prediction:")
X, t02, t_max2 = plot_linear({t:v for t,v in values_diff.items() if t-t0>10*3600*24 and tmax-t > 4*24*3600},3,"diff(log(total_deaths)) last but 4 days")

T_stop = -X[1]/X[0] 
print(f"T_stop={T_stop}")
print("timestamp = ",str(t02+3600*24*T_stop))
d = datetime.fromtimestamp(t02+3600*24*T_stop)
print("Date = ", d.ctime())

T_last_sec = t_max2
T_last_days = (T_last_sec - t02)/(24*3600)
prognosis_T_last = X[0] * T_last_days + X[1]
delta_log_total_deaths = prognosis_T_last * (T_stop-T_last_days)/2.0
log_total_deaths_prognosed = values[T_last_sec][3] + delta_log_total_deaths
total_deaths_prognosed = exp(log_total_deaths_prognosed)
print(f"total_deaths_prognosed = {total_deaths_prognosed}")

#
# Let's see if this works for total number of cases
#
X, t01, t_max1 = plot_linear({t:v for t,v in values_diff.items() if t-t0>10*3600*24},2,"diff(log(total)) last")
#
# Let's try to extrapolate this
#

if "--show" in argv:
    _ = plt.plot(range(100),
        [ X[0]*t+X[1] for t in range(100)], 'r', label="prognosis(diff(log(total)))")
    plt.show()
#
# Ok. This means that there is a decline and we expect to have 0 new cases at:
#
T_stop = -X[1]/X[0] 
print("Max-precision prediction (total cases):")
print(f"T_stop={T_stop} [days]")
print("timestamp = ",str(t01+3600*24*T_stop))
d = datetime.fromtimestamp(t01+3600*24*T_stop)
print("Date = ", d.ctime())

T_last_sec = t_max1
T_last_days = (T_last_sec - t01)/(24*3600)
prognosis_T_last = X[0] * T_last_days + X[1]
delta_log_total = prognosis_T_last * (T_stop-T_last_days)/2.0
log_total_prognosed = values[T_last_sec][2] + delta_log_total
total_prognosed = exp(log_total_prognosed)
print(f"total_prognosed = {total_prognosed}")


X, t01, t_max1 = plot_linear({t:v for t,v in values_diff.items() if t-t0>10*3600*24 and tmax-t > 2*24*3600 },2,"diff(log(total)) last")

T_stop = -X[1]/X[0] 
print("2-days-removed prediction (total cases):")
print(f"T_stop={T_stop} [days]")
print("timestamp = ",str(t01+3600*24*T_stop))
d = datetime.fromtimestamp(t01+3600*24*T_stop)
print("Date = ", d.ctime())

T_last_sec = t_max1
T_last_days = (T_last_sec - t01)/(24*3600)
prognosis_T_last = X[0] * T_last_days + X[1]
delta_log_total = prognosis_T_last * (T_stop-T_last_days)/2.0
log_total_prognosed = values[T_last_sec][2] + delta_log_total
total_prognosed = exp(log_total_prognosed)
print(f"total_prognosed = {total_prognosed}")


