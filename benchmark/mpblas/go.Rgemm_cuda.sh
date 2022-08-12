#!/bin/bash

if [ `uname` = "Darwin" ]; then
    LDPATHPREFIX="DYLD_LIBRARY_PATH=%%PREFIX%%/lib"
else
    LDPATHPREFIX="LD_LIBRARY_PATH=%%PREFIX%%/lib:$LD_LIBRARY_PATH"
fi

####

env $LDPATHPREFIX ./Rgemm.dd_cuda_kernel -NOCHECK -TOTALSTEPS 700 -STEPK 5 -STEPM 5 -STEPN 5 -LOOP 5   >& log.Rgemm.dd_cuda_kernel
env $LDPATHPREFIX ./Rgemm.dd_cuda_total  -NOCHECK -TOTALSTEPS 700 -STEPK 5 -STEPM 5 -STEPN 5 -LOOP 3   >& log.Rgemm.dd_cuda_total

gnuplot Rgemm_cuda.plt > Rgemm_cuda.pdf
