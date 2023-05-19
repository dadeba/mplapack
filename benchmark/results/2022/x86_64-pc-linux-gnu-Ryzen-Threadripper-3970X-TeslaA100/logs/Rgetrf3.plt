set xlabel font "Helvetica,20"
set ylabel font "Helvetica,20"
set key font "Helvetica,16"
set title font "Helvetica,20"
set title "Rgetrf on AMD Ryzen Threadripper 3970X 32-Core Processor "
set xlabel "Dimension"
set ylabel "MFLOPS"
set key above
set terminal pdf

plot \
"log.Rgetrf.double_opt"	    using 1:3 title 'double (OpenMP)'        with lines linewidth 1, \
"log.Rgetrf.double"         using 1:3 title 'double'                 with lines linewidth 1, \
"log.dgetrf.ref"	    using 1:3 title 'double (Ref.BLAS)'      with lines linewidth 1, \
"log.dgetrf.openblas"       using 1:3 title 'double (OpenBLAS)'      with lines linewidth 1
