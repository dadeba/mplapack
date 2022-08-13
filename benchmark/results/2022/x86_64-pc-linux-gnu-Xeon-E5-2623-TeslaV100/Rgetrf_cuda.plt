set xlabel font "Helvetica,20"
set ylabel font "Helvetica,20"
set key font "Helvetica,16"
set title font "Helvetica,24"
set title "Rgetrf on NVIDIA Tesla V100 PCIe 32GB (rev a1)"
set xlabel "Dimension"
set ylabel "MFLOPS"
#set terminal postscript eps color enhanced
set terminal pdf

plot "log.Rgetrf.dd_cuda_total"  using 1:3 title 'Total ' with lines linewidth 6
