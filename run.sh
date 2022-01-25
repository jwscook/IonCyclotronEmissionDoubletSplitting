#!/bin/bash
source ~/.bashrc
shopt -s expand_aliases

# first arg is pitch, second arg is ring thermal spread as fraction of its speed
julia1.7 -e 'using Pkg; Pkg.instantiate()'
julia1.7 --proj ICE2D.jl -0.64 0.01 0.01 physical
julia1.7 --proj ICE2D.jl  0.0  0.01 0.01 ring

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
