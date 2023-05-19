set xlabel font "Helvetica,20"
set ylabel font "Helvetica,20"
set key font "Helvetica,16"
set title font "Helvetica,20"
set title "Rgetrf on NVIDIA A100 80GB PCIe"
set xlabel "Dimension"
set ylabel "MFLOPS"
set key above
set terminal pdf

plot "log.Rgetrf.dd_cuda_total"  using 1:3 title 'Total' with lines linewidth 1
