#!/bin/bash

USE_CCACHE=yes

if [ x$USE_CCACHE = x"yes" ] ; then
CXX="ccache icpc" ; export CXX
CC="ccache icc" ; export CC
FC="ifort"; export FC
F77="ifort"; export F77
ccache -M 20G
else
CXX="icpc" ; export CXX
CC="icc" ; export CC
FC="ifort"; export FC
F77="ifort"; export F77
fi

_GCC="gcc-9"; export _GCC
_GXX="g++-9"; export _GXX
_GFC="gfortran-9"; export _GFC

pushd mplapack/debug ; bash gen.Makefile.am.sh ; popd

aclocal ; autoconf ; automake --add-missing
autoreconf --force --install

./configure --prefix=$HOME/MPLAPACK --enable-gmp=yes --enable-mpfr=yes --enable-_Float128=yes --enable-qd=yes --enable-dd=yes --enable-double=yes --enable-_Float64x=yes --enable-debug=yes