set xlabel font "Helvetica,20"
set ylabel font "Helvetica,20"
set key font "Helvetica,16"
set title font "Helvetica,24"
set title "Rdot on Intel(R) Xeon(R) CPU E5-2623 v3 @ 3.00GHz "
set xlabel "Dimension"
set ylabel "MFLOPS"
#set terminal postscript eps color enhanced
set terminal pdf

plot \
"log.Rdot.mpfr"            using 1:2 title 'MPFR 512bit'            with lines linewidth 2, \
"log.Rdot.mpfr_opt"        using 1:2 title 'MPFR 512bit(OpenMP)'    with lines linewidth 2, \
"log.Rdot.gmp"             using 1:2 title 'GMP  512bit'            with lines linewidth 2, \
"log.Rdot.gmp_opt"         using 1:2 title 'GMP  512bit(OpenMP)'    with lines linewidth 2, \
"log.Rdot.qd"              using 1:2 title 'quad-double'            with lines linewidth 2, \
"log.Rdot.qd_opt"          using 1:2 title 'quad-double(OpenMP)'    with lines linewidth 2
