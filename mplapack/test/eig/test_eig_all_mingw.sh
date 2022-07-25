#!/bin/bash

export WINEPATH="/usr/x86_64-w64-mingw32/lib/;/usr/lib/gcc/x86_64-w64-mingw32/9.3-win32/;/usr/lib/gcc/x86_64-w64-mingw32/9.3-posix;/home/docker/MPLAPACK_MINGW/bin"
export WINEDEBUG="-all"
JOBS=4

EIGREALS=`ls *xeigtstR_* | grep -v log`
EIGCOMPLEXES=`ls *xeigtstC_* | grep -v log`

TESTREALS=`ls R*.in [a-z]*.in  | grep -v double | grep -v ^log`
TESTCOMPLEXES=`ls C*.in [a-z]*.in | grep -v double | grep -v ^log`

rm -f .parallel.test_eig_all_mingw.sh

echo "/usr/bin/time wine64 ./*xeigtstR_double.exe < Rbal_double.in >& log.x86_64-w64-mingw32-xeigtstR_double.Rbal.in" >> .parallel.test_eig_all_mingw.sh
echo "/usr/bin/time wine64 ./*xeigtstC_double.exe < Cbal_double.in >& log.x86_64-w64-mingw32-xeigtstC_double.Cbal.in"  >> .parallel.test_eig_all_mingw.sh

for eigreal in $EIGREALS; do
    for testreal in $TESTREALS; do
        echo "/usr/bin/time wine64 ./$eigreal < $testreal >& log.$eigreal.$testreal" >> .parallel.test_eig_all_mingw.sh
    done
done

for eigcomplex in $EIGCOMPLEXES; do
    for testcomplex in $TESTCOMPLEXES; do
        echo "/usr/bin/time wine64 ./$eigcomplex < $testcomplex >& log.$eigcomplex.$testcomplex" >> .parallel.test_eig_all_mingw.sh
    done
done

cat .parallel.test_eig_all_mingw.sh | parallel --jobs $JOBS

rm .parallel.test_eig_all_mingw.sh
