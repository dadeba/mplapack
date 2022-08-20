set xlabel font "Helvetica,20"
set ylabel font "Helvetica,20"
set key font "Helvetica,16"
set title font "Helvetica,20"
set title "Rcopy on %%MODELNAME%%"
set xlabel "Dimension"
set ylabel "MFLOPS"
#set terminal postscript eps color enhanced
set terminal pdf

plot \
"log.Rcopy.dd_opt"          using 1:2 title 'double double (OpenMP)' with lines linewidth 1, \
"log.Rcopy._Float128"       using 1:2 title '\_Float128'             with lines linewidth 1, \
"log.Rcopy._Float128_opt"   using 1:2 title '\_Float128 (OpenMP)'    with lines linewidth 1, \
"log.Rcopy._Float64x"       using 1:2 title '\_Float64x'             with lines linewidth 1, \
"log.Rcopy._Float64x_opt"   using 1:2 title '\_Float64x (OpenMP)'    with lines linewidth 1, \
"log.Rcopy.dd"              using 1:2 title 'double double'          with lines linewidth 1, \
"log.Rcopy.double"          using 1:2 title 'double'                 with lines linewidth 1
