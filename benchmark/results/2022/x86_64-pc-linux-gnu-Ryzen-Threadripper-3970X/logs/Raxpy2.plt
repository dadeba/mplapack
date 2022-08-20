set xlabel font "Helvetica,20"
set ylabel font "Helvetica,20"
set key font "Helvetica,16"
set title font "Helvetica,24"
set title "Raxpy on AMD Ryzen Threadripper 3970X 32-Core Processor "
set xlabel "Dimension"
set ylabel "MFLOPS"
#set terminal postscript eps color enhanced
set terminal pdf

plot \
"log.Raxpy.mpfr"            using 1:2 title 'MPFR 512bit'            with lines linewidth 1, \
"log.Raxpy.mpfr_opt"        using 1:2 title 'MPFR 512bit(OpenMP)'    with lines linewidth 1, \
"log.Raxpy.gmp"             using 1:2 title 'GMP  512bit'            with lines linewidth 1, \
"log.Raxpy.gmp_opt"         using 1:2 title 'GMP  512bit(OpenMP)'    with lines linewidth 1, \
"log.Raxpy.qd"              using 1:2 title 'quad-double'            with lines linewidth 1, \
"log.Raxpy.qd_opt"	    using 1:2 title 'quad-double(OpenMP)'    with lines linewidth 1
