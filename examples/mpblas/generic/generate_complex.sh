FILES=`ls C*generic.cpp`
MPLIBS="mpfr gmp _Float128 _Float64x double dd qd"

for _file in $FILES; do
    for _mplib in $MPLIBS; do
        resultfilename=`echo $_file | sed "s/generic/${_mplib}/g"`
        cat ../../generic/header_${_mplib}_complex ${_file} > ../$resultfilename
        if [ x"$_mplib" = x"gmp" ]; then
            sed -i -e "s/REAL/mpf_class/g" -e "s/COMPLEX/mpc_class/g" -e "s/INTEGER/mplapackint/g" -e "s/InTEGER/INTEGER_${_mplib}/g" -e "s/ReAL/REAL_${_mplib}/g" -e "s/Rlamch/Rlamch_${_mplib}/g"  ../$resultfilename
        fi
        if [ x"$_mplib" = x"mpfr" ]; then
            sed -i -e "s/REAL/mpreal/g" -e "s/COMPLEX/mpcomplex/g" -e "s/INTEGER/mplapackint/g" -e "s/InTEGER/INTEGER_${_mplib}/g" -e "s/ReAL/REAL_${_mplib}/g" -e "s/Rlamch/Rlamch_${_mplib}/g" ../$resultfilename
        fi
        if [ x"$_mplib" = x"double" ]; then
            sed -i -e "s/REAL/double/g" -e "s/COMPLEX/std::complex<double>/g" -e "s/INTEGER/mplapackint/g" -e "s/InTEGER/INTEGER_${_mplib}/g" -e "s/ReAL/REAL_${_mplib}/g" -e "s/Rlamch/Rlamch_${_mplib}/g" ../$resultfilename
        fi 
        if [ x"$_mplib" = x"dd" ]; then
            sed -i -e "s/REAL/dd_real/g" -e "s/COMPLEX/dd_complex/g" -e "s/INTEGER/mplapackint/g" -e "s/InTEGER/INTEGER_${_mplib}/g" -e "s/ReAL/REAL_${_mplib}/g" -e "s/Rlamch/Rlamch_${_mplib}/g" ../$resultfilename
        fi 
        if [ x"$_mplib" = x"qd" ]; then
            sed -i -e "s/REAL/qd_real/g" -e "s/COMPLEX/qd_complex/g" -e "s/INTEGER/mplapackint/g" -e "s/InTEGER/INTEGER_${_mplib}/g" -e "s/ReAL/REAL_${_mplib}/g" -e "s/Rlamch/Rlamch_${_mplib}/g" ../$resultfilename
        fi
        if [ x"$_mplib" = x"_Float128" ]; then
            sed -i -e "s/REAL/_Float128/g" -e "s/COMPLEX/std::complex<_Float128>/g" -e "s/INTEGER/mplapackint/g" -e "s/InTEGER/INTEGER_${_mplib}/g" -e "s/ReAL/REAL_${_mplib}/g" -e "s/Rlamch/Rlamch_${_mplib}/g" ../$resultfilename
        fi
        if [ x"$_mplib" = x"_Float64x" ]; then
            sed -i -e "s/REAL/_Float64x/g" -e "s/COMPLEX/std::complex<_Float64x>/g" -e "s/INTEGER/mplapackint/g" -e "s/InTEGER/INTEGER_${_mplib}/g" -e "s/ReAL/REAL_${_mplib}/g" -e "s/Rlamch/Rlamch_${_mplib}/g" ../$resultfilename
        fi
    done
done

FILES=`ls *generic.cpp`

echo "mplapackexamples_PROGRAMS =" > ../Makefile.am
for _mplib in $MPLIBS; do

    if [ x"$_mplib" = x"mpfr" ]; then
        echo ""               >> ../Makefile.am   
        echo "if ENABLE_MPFR" >> ../Makefile.am
        executefilenames=`echo $FILES | sed 's/\.cpp//g' | sed "s/generic/${_mplib}/g"`
        echo "mplapackexamples_PROGRAMS += $executefilenames" >> ../Makefile.am
        echo ""               >> ../Makefile.am   
        echo "${_mplib}_cxxflags = \$(OPENMP_CXXFLAGS) -I\$(top_srcdir)/include -I\$(top_srcdir)/mpfrc++ -I\$(GMP_INCLUDEDIR) -I\$(MPFR_INCLUDEDIR) -I\$(MPC_INCLUDEDIR)" >> ../Makefile.am
        echo "${_mplib}_libdepends = -L\$(top_builddir)/mplapack/reference -lmplapack_${_mplib} -L\$(top_builddir)/mpblas/reference -lmpblas_${_mplib} -L\$(MPC_LIBDIR) -L\$(MPFR_LIBDIR) -L\$(GMP_LIBDIR) -lmpfr -lmpc -lgmp"  >> ../Makefile.am   
        echo ""               >> ../Makefile.am
        for _file in $FILES; do
            A=`echo $_file | sed "s/generic\.cpp/${_mplib}/g"` 
            echo "${A}_SOURCES = ${A}.cpp" >> ../Makefile.am
            echo "${A}_CXXFLAGS = \$(${_mplib}_cxxflags)" >> ../Makefile.am
            echo "${A}_LDFLAGS = \$(${_mplib}_libdepends)" >> ../Makefile.am
            echo ""               >> ../Makefile.am
        done
        echo "endif"             >> ../Makefile.am   
    fi

    if [ x"$_mplib" = x"gmp" ]; then
        echo ""               >> ../Makefile.am   
        echo "if ENABLE_GMP" >> ../Makefile.am
        executefilenames=`echo $FILES | sed 's/\.cpp//g' | sed "s/generic/${_mplib}/g"`
        echo "mplapackexamples_PROGRAMS += $executefilenames" >> ../Makefile.am
        echo ""               >> ../Makefile.am   
        echo "${_mplib}_cxxflags = \$(OPENMP_CXXFLAGS) -I\$(top_srcdir)/include -I\$(GMP_INCLUDEDIR)" >> ../Makefile.am
        echo "${_mplib}_libdepends = -L\$(top_builddir)/mplapack/reference -lmplapack_${_mplib} -L\$(top_builddir)/mpblas/reference -lmpblas_${_mplib} -L\$(GMP_LIBDIR) -lgmp"  >> ../Makefile.am   
        echo ""               >> ../Makefile.am
        for _file in $FILES; do
            A=`echo $_file | sed "s/generic\.cpp/${_mplib}/g"` 
            echo "${A}_SOURCES = ${A}.cpp" >> ../Makefile.am
            echo "${A}_CXXFLAGS = \$(${_mplib}_cxxflags)" >> ../Makefile.am
            echo "${A}_LDFLAGS = \$(${_mplib}_libdepends)" >> ../Makefile.am
            echo ""               >> ../Makefile.am
        done
        echo "endif"             >> ../Makefile.am   
    fi

    if [ x"$_mplib" = x"_Float128" ]; then
        echo ""               >> ../Makefile.am   
        echo "if ENABLE__FLOAT128" >> ../Makefile.am
        executefilenames=`echo $FILES | sed 's/\.cpp//g' | sed "s/generic/${_mplib}/g"`
        echo "mplapackexamples_PROGRAMS += $executefilenames" >> ../Makefile.am
        echo ""               >> ../Makefile.am   
        echo "${_mplib}_cxxflags = \$(OPENMP_CXXFLAGS)" >> ../Makefile.am
        echo "if WANT_QUADMATH" >> ../Makefile.am
        echo "${_mplib}_libdepends = -L\$(top_builddir)/mplapack/reference -lmplapack_${_mplib} -L\$(top_builddir)/mpblas/reference -lmpblas_${_mplib} -lquadmath"  >> ../Makefile.am   
        echo "else" >> ../Makefile.am
        echo "${_mplib}_libdepends = -L\$(top_builddir)/mplapack/reference -lmplapack_${_mplib} -L\$(top_builddir)/mpblas/reference -lmpblas_${_mplib}"  >> ../Makefile.am   
        echo "endif" >> ../Makefile.am
        echo ""               >> ../Makefile.am
        for _file in $FILES; do
            A=`echo $_file | sed "s/generic\.cpp/${_mplib}/g"` 
            echo "${A}_SOURCES = ${A}.cpp" >> ../Makefile.am
            echo "${A}_CXXFLAGS = \$(${_mplib}_cxxflags)" >> ../Makefile.am
            echo "${A}_LDFLAGS = \$(${_mplib}_libdepends)" >> ../Makefile.am
            echo ""               >> ../Makefile.am
        done
        echo "endif"             >> ../Makefile.am   
    fi

    if [ x"$_mplib" = x"_Float64x" ]; then
        echo ""               >> ../Makefile.am   
        echo "if ENABLE__FLOAT64X" >> ../Makefile.am
        executefilenames=`echo $FILES | sed 's/\.cpp//g' | sed "s/generic/${_mplib}/g"`
        echo "mplapackexamples_PROGRAMS += $executefilenames" >> ../Makefile.am
        echo ""               >> ../Makefile.am   
        echo "${_mplib}_cxxflags = \$(OPENMP_CXXFLAGS)" >> ../Makefile.am
        echo "${_mplib}_libdepends = -L\$(top_builddir)/mplapack/reference -lmplapack_${_mplib} -L\$(top_builddir)/mpblas/reference -lmpblas_${_mplib}"  >> ../Makefile.am   
        echo ""               >> ../Makefile.am
        for _file in $FILES; do
            A=`echo $_file | sed "s/generic\.cpp/${_mplib}/g"` 
            echo "${A}_SOURCES = ${A}.cpp" >> ../Makefile.am
            echo "${A}_CXXFLAGS = \$(${_mplib}_cxxflags)" >> ../Makefile.am
            echo "${A}_LDFLAGS = \$(${_mplib}_libdepends)" >> ../Makefile.am
            echo ""               >> ../Makefile.am
        done
        echo "endif"             >> ../Makefile.am   
    fi

    if [ x"$_mplib" = x"double" ]; then
        echo ""               >> ../Makefile.am   
        echo "if ENABLE_DOUBLE" >> ../Makefile.am
        executefilenames=`echo $FILES | sed 's/\.cpp//g' | sed "s/generic/${_mplib}/g"`
        echo "mplapackexamples_PROGRAMS += $executefilenames" >> ../Makefile.am
        echo ""               >> ../Makefile.am   
        echo "${_mplib}_cxxflags = \$(OPENMP_CXXFLAGS)" >> ../Makefile.am
        echo "${_mplib}_libdepends = -L\$(top_builddir)/mplapack/reference -lmplapack_${_mplib} -L\$(top_builddir)/mpblas/reference -lmpblas_${_mplib}"  >> ../Makefile.am   
        echo ""               >> ../Makefile.am
        for _file in $FILES; do
            A=`echo $_file | sed "s/generic\.cpp/${_mplib}/g"` 
            echo "${A}_SOURCES = ${A}.cpp" >> ../Makefile.am
            echo "${A}_CXXFLAGS = \$(${_mplib}_cxxflags)" >> ../Makefile.am
            echo "${A}_LDFLAGS = \$(${_mplib}_libdepends)" >> ../Makefile.am
            echo ""               >> ../Makefile.am
        done
        echo "endif"             >> ../Makefile.am   
    fi

    if [ x"$_mplib" = x"dd" ]; then
        echo ""               >> ../Makefile.am   
        echo "if ENABLE_DD" >> ../Makefile.am
        executefilenames=`echo $FILES | sed 's/\.cpp//g' | sed "s/generic/${_mplib}/g"`
        echo "mplapackexamples_PROGRAMS += $executefilenames" >> ../Makefile.am
        echo ""               >> ../Makefile.am   
        echo "${_mplib}_cxxflags = \$(OPENMP_CXXFLAGS) -I\$(top_srcdir)/include -I\$(QD_INCLUDEDIR)" >> ../Makefile.am
        echo "${_mplib}_libdepends = -L\$(top_builddir)/mplapack/reference -lmplapack_${_mplib} -L\$(top_builddir)/mpblas/reference -lmpblas_${_mplib} -L\$(QD_LIBDIR) -lqd"  >> ../Makefile.am   
        echo ""               >> ../Makefile.am
        for _file in $FILES; do
            A=`echo $_file | sed "s/generic\.cpp/${_mplib}/g"` 
            echo "${A}_SOURCES = ${A}.cpp" >> ../Makefile.am
            echo "${A}_CXXFLAGS = \$(${_mplib}_cxxflags)" >> ../Makefile.am
            echo "${A}_LDFLAGS = \$(${_mplib}_libdepends)" >> ../Makefile.am
            echo ""               >> ../Makefile.am
        done
        echo "endif"             >> ../Makefile.am   
    fi

    if [ x"$_mplib" = x"qd" ]; then
        echo ""               >> ../Makefile.am   
        echo "if ENABLE_QD" >> ../Makefile.am
        executefilenames=`echo $FILES | sed 's/\.cpp//g' | sed "s/generic/${_mplib}/g"`
        echo "mplapackexamples_PROGRAMS += $executefilenames" >> ../Makefile.am
        echo ""               >> ../Makefile.am   
        echo "${_mplib}_cxxflags = \$(OPENMP_CXXFLAGS) -I\$(top_srcdir)/include -I\$(QD_INCLUDEDIR)" >> ../Makefile.am
        echo "${_mplib}_libdepends = -L\$(top_builddir)/mplapack/reference -lmplapack_${_mplib} -L\$(top_builddir)/mpblas/reference -lmpblas_${_mplib} -L\$(QD_LIBDIR) -lqd"  >> ../Makefile.am   
        echo ""               >> ../Makefile.am
        for _file in $FILES; do
            A=`echo $_file | sed "s/generic\.cpp/${_mplib}/g"` 
            echo "${A}_SOURCES = ${A}.cpp" >> ../Makefile.am
            echo "${A}_CXXFLAGS = \$(${_mplib}_cxxflags)" >> ../Makefile.am
            echo "${A}_LDFLAGS = \$(${_mplib}_libdepends)" >> ../Makefile.am
            echo ""               >> ../Makefile.am
        done
        echo "endif"             >> ../Makefile.am   
    fi

done
echo ""               >> ../Makefile.am
path=`pwd` 
array=( `echo $path | tr -s '/' ' '`)
kind_index=`expr ${#array[@]} - 2`
echo "mplapackexamplesdir=\$(prefix)/share/mplapack/examples/${array[${kind_index}]}"   >> ../Makefile.am   