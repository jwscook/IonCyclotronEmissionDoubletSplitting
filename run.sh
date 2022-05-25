#!/bin/bash

julia --proj=@. -e 'using Pkg; Pkg.instantiate(); ENV["PYTHON"]=""; Pkg.build("PyCall"); using Conda;Conda.rm("matplotlib"); Conda.add("matplotlib=3.3.4")'
# first arg is pitch, second arg is ring thermal spread as fraction of its speed
julia --proj=@. ICE2D.jl -0.646 0.01 0.01 physical
julia --proj=@. ICE2D.jl  0.0   0.01 0.01 ring

# now reduce size and improve images
for j in physical ring
do
  for i in `ls *$j.pdf`
  do
    pdf2ps $i $i.ps
    ps2pdf $i.ps $i
    rm $i.ps
  done
done

cp ICE2D_Combo_ring.pdf fig1.pdf
cp ICE2D_imag_ring.pdf fig2.pdf
cp ICE2D_Combo_physical.pdf fig3.pdf
cp ICE2D_imag_physical.pdf fig4.pdf
cp ICE2D_real_div_guess_ring.pdf fig5.pdf

pdflatex manuscript.tex
pdflatex manuscript.tex
