set xlabel font "Helvetica,20"
set ylabel font "Helvetica,20"
set key font "Helvetica,16"
set title font "Helvetica,20"
set title "Rgemm on %%MODELNAME%%"
set xlabel "Dimension"
set ylabel "MFLOPS"
#set terminal postscript eps color enhanced
set terminal pdf

plot \
"log.Rgemm.mpfr"            using 1:4 title 'MPFR 512bit'            with lines linewidth 1, \
"log.Rgemm.mpfr_opt"        using 1:4 title 'MPFR 512bit(OpenMP)'    with lines linewidth 1, \
"log.Rgemm.gmp"             using 1:4 title 'GMP  512bit'            with lines linewidth 1, \
"log.Rgemm.gmp_opt"         using 1:4 title 'GMP  512bit(OpenMP)'    with lines linewidth 1, \
"log.Rgemm.qd"              using 1:4 title 'quad-double'            with lines linewidth 1, \
"log.Rgemm.qd_opt"	    using 1:4 title 'quad-double(OpenMP)'    with lines linewidth 1
