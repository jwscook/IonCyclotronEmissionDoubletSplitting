
using Distributed, Dates
using Plots, Random, ImageFiltering, Statistics
using Dierckx, Contour, JLD2, DelimitedFiles

println("Starting at ", now())
# pitchanglecosine of 0 is all vperp, and (-)1 is (anti-)parallel to B field
# pitch angle is defined as vpara/v0
const pitchanglecosine = try; parse(Float64, ARGS[1]); catch; -0.646; end
@assert -1 <= pitchanglecosine <= 1
# thermal width of ring as a fraction of its speed # Dendy PRL 1993
const vthermalfractionz = try; parse(Float64, ARGS[2]); catch; 0.01; end
const vthermalfraction⊥ = try; parse(Float64, ARGS[3]); catch; 0.01; end
# secondary fuel concentration
const xi2 = try; parse(Float64, ARGS[4]); catch; 0.0; end
# name of file
const name_extension = if length(ARGS) >= 5
  ARGS[5]
else
  "$(pitchanglecosine)_$(vthermalfractionz)_$(vthermalfraction⊥)"
end
const filecontents = [i for i in readlines(open(@__FILE__))]
const nprocsadded = div(Sys.CPU_THREADS, 2)

addprocs(nprocsadded, exeflags="--project")

@everywhere using ProgressMeter # for some reason must be up here on its own
@everywhere using StaticArrays
@everywhere using FastClosures
@everywhere begin
  using LinearMaxwellVlasov, LinearAlgebra, WindingNelderMead

  # mass ratios 
  const mp = 1836.2
  const md = 3671.5
  const mT = 5497.93
  const mHe3 = 5497.885
  const mα = 7294.3

  mₑ = LinearMaxwellVlasov.mₑ
  m1 = mp*mₑ
  m2 = 0 #mT*mₑ
  mmin = mₑ #mα*mₑ #mp*mₑ
  
  ze = -1
  z1 = 1
  z2 = 0
  zmin = -1
  
  # Fig 18 Cottrell 1993
  n0 = 1.5e19 #1e19# 5e19 # 1.7e19 # central electron density 3.6e19
  B0 = 1.9 #2.1 #2.07 = 2.8T * 2.96 m / 4m
  # 2.23 T is 17MHz for deuterium cyclotron frequency
  ξ = 1e-2#1.5e-4 # nα / ni = 1.5 x 10^-4
  ξ2 = Float64(@fetchfrom 1 xi2) # 0.15
  n2 = ξ2*n0
  nmin = ξ*n0
  n1 = (1/z1)*(n0-z2*n2-zmin*nmin) # 1 / (1.0 + 2*ξ)
  @assert n0 ≈ z1*n1 + z2*n2 + zmin*nmin
  density_weighted = n1*m1 + n2*m2 + nmin*mmin
  Va = B0 / sqrt(LinearMaxwellVlasov.μ₀*density_weighted)

  Te = 3e3# eV
  T1 = Te # eV
  T2 = T1 # eV
  Emin = 100e3 #3.5e6 # eV # 14.67e6
  Ωe = cyclotronfrequency(B0, mₑ, ze)
  Ω1 = cyclotronfrequency(B0, m1, z1)
  Ωmin = cyclotronfrequency(B0, mmin, zmin)
  Πe = plasmafrequency(n0, mₑ, ze)
  Π1 = plasmafrequency(n1, m1, z1)
  Πmin = plasmafrequency(nmin, mmin, zmin)
  vthe = thermalspeed(Te, mₑ) # temperature, mass
  vth1 = thermalspeed(T1, m1) # temperature, mass
  vmin = thermalspeed(Emin, mmin) # energy in terms of eV (3.5e6)
  if m2 != 0
    Ω2 = cyclotronfrequency(B0, m2, z2)
    Π2 = plasmafrequency(n2, m2, z2)
    vth2 = thermalspeed(T2, m2) # temperature, mass
    spec2_cold = ColdSpecies(Π2, Ω2)
    spec2_warm = WarmSpecies(Π2, Ω2, vth2)
    spec2_maxw = MaxwellianSpecies(Π2, Ω2, vth2, vth2)  
  else
    Ω2 = 0
    Π2 = 0
    vth2 = 0
    spec2_cold = 0
    spec2_warm = 0
    spec2_maxw = 0
  end
  # pitchanglecosine = cos(pitchangle)
  # acos(pitchanglecosine) = pitchangle
  pitchanglecosine = Float64(@fetchfrom 1 pitchanglecosine)
  vα⊥ = vmin * sqrt(1 - pitchanglecosine^2) # perp speed
  vαz = vmin * pitchanglecosine # parallel speed
  vthermalfractionz = Float64(@fetchfrom 1 vthermalfractionz)
  vthermalfraction⊥ = Float64(@fetchfrom 1 vthermalfraction⊥)
  vαthz = vmin * vthermalfractionz
  vαth⊥ = vmin * vthermalfraction⊥

  electron_cold = ColdSpecies(Πe, Ωe)
  electron_warm = WarmSpecies(Πe, Ωe, vthe)
  electron_maxw = MaxwellianSpecies(Πe, Ωe, vthe, vthe)

  spec1_cold = ColdSpecies(Π1, Ω1)
  spec1_warm = WarmSpecies(Π1, Ω1, vth1)
  spec1_maxw = MaxwellianSpecies(Π1, Ω1, vth1, vth1)

  minspec_cold = ColdSpecies(Πmin, Ωmin)
  minspec_maxw = MaxwellianSpecies(Πmin, Ωmin, vαthz, vαth⊥, vαz)
  minspec_ringbeam = SeparableVelocitySpecies(Πmin, Ωmin,
    FBeam(vαthz, vαz),
    FRing(vαth⊥, vα⊥))
  minspec_delta = SeparableVelocitySpecies(Πmin, Ωmin,
    FParallelDiracDelta(vαz),
    FPerpendicularDiracDelta(vα⊥))

  if m2 != 0
    Smmr = Plasma([electron_maxw, spec1_maxw, spec2_maxw, minspec_ringbeam]) #spec2_maxw change these for multiple ions
    Smmd = Plasma([electron_maxw, spec1_maxw, spec2_maxw, minspec_delta]) # 
  else
    Smmr = Plasma([electron_maxw, spec1_maxw, spec2_maxw, minspec_ringbeam]) #spec2_maxw change these for multiple ions
    Smmd = Plasma([electron_maxw, spec1_maxw, spec2_maxw, minspec_delta]) # 
  end

  w0 = abs(Ωmin)
  k0 = w0 / abs(Va)

  γmax = abs(Ωmin) * 0.15
  γmin = -abs(Ωmin) * 0.075
  function bounds(ω0)
    lb = @SArray [ω0 * 0.5, γmin]
    ub = @SArray [ω0 * 1.2, γmax]
    return (lb, ub)
  end

  options = Options(memoiseparallel=false, memoiseperpendicular=true)

  function solve_given_ks(K, objective!)
    ω0 = fastzerobetamagnetoacousticfrequency(Va, K, Ω1)

    lb, ub = bounds(ω0)

    function boundify(f::T) where {T}
      @inbounds isinbounds(x) = all(i->0 <= x[i] <= 1, eachindex(x))
      maybeaddinf(x::U, addinf::Bool) where {U} = addinf ? x + U(Inf) : x
      bounded(x) = maybeaddinf(f(x), !isinbounds(x))
      return bounded
    end

    config = Configuration(K, options)

    ics = ((@SArray [ω0*0.8, w0*0.08]),
           (@SArray [ω0*0.9, w0*0.04]),
           (@SArray [ω0*1.0, w0*0.01]))


    function unitobjective!(c, x::T) where {T}
      return objective!(c,
        T([x[i] * (ub[i] - lb[i]) + lb[i] for i in eachindex(x)]))
    end
    unitobjectivex! = x -> unitobjective!(config, x)
    boundedunitobjective! = boundify(unitobjectivex!)
    xtol_abs = w0 .* (@SArray [1e-4, 1e-5]) ./ (ub .- lb)
    @elapsed for ic ∈ ics
      @assert all(i->lb[i] <= ic[i] <= ub[i], eachindex(ic))
      neldermeadsol = WindingNelderMead.optimise(
        boundedunitobjective!, SArray((ic .- lb) ./ (ub .- lb)),
        1.1e-2 * (@SArray ones(2)); stopval=1e-15, timelimit=30,
        maxiters=200, ftol_rel=0, ftol_abs=0, xtol_rel=0, xtol_abs=xtol_abs)
      simplex, windingnumber, returncode, numiterations = neldermeadsol
      if (windingnumber == 1 && returncode == :XTOL_REACHED)# || returncode == :STOPVAL_REACHED
        c = deepcopy(config)
        minimiser = if windingnumber == 0
          WindingNelderMead.position(WindingNelderMead.bestvertex(simplex))
        else
          WindingNelderMead.centre(simplex)
        end
        unitobjective!(c, minimiser)
        return c
      end
    end
    return nothing
  end

  function f2Dω!(config::Configuration, x::AbstractArray, plasma, cache)
    config.frequency = Complex(x[1], x[2])
    return det(tensor(plasma, config, cache))
  end

  function findsolutions(plasma)
    ngridpoints = 2^9
    kzs = range(-2.0, stop=2.0, length=ngridpoints) * k0
    k⊥s = range(0.0, stop=15.0, length=ngridpoints) * k0

    # change order for better distributed scheduling
    k⊥s = shuffle(vcat([k⊥s[i:nprocs():end] for i ∈ 1:nprocs()]...))
    @assert length(k⊥s) == ngridpoints
    solutions = @sync @showprogress @distributed (vcat) for k⊥ ∈ k⊥s
      cache = Cache()
      objective! = @closure (C, x) -> f2Dω!(C, x, plasma, cache)
      innersolutions = Vector()
      for (ikz, kz) ∈ enumerate(kzs)
        K = Wavenumber(parallel=kz, perpendicular=k⊥)
        output = solve_given_ks(K, objective!)
        isnothing(output) && continue
        push!(innersolutions, output)
      end
      innersolutions
    end
    return solutions
  end
end #@everywhere

"Select the largest growth rates if multiple solutions found for a wavenumber"
function selectlargeestgrowthrate(sols)
    imagfreq(s) = imag(s.frequency)
    d = Dict{Any, Vector{eltype(sols)}}()
    for sol in sols
      if haskey(d, sol.wavenumber)
        push!(d[sol.wavenumber], sol)
      else
        d[sol.wavenumber] = [sol]
      end
    end
    output = Vector()
    sizehint!(output, length(sols))
    for (_, ss) in d
      push!(output, ss[findmax(map(imagfreq, ss))[2]])
    end
    return output
end

function selectpropagationrange(sols, lowangle=0, highangle=180)
  function propangle(s)
    kz = para(s.wavenumber)
    k⊥ = perp(s.wavenumber)
    θ = atan.(k⊥, kz) * 180 / pi
  end
  output = Vector()
  for s in sols
    (lowangle <= propangle(s) <= highangle) || continue
    push!(output, s)
  end
  return output
end

Plots.gr() # .pyplot()
function plotit(sols, file_extension=name_extension, fontsize=9)
  sols = sort(sols, by=s->imag(s.frequency))
  ωs = [sol.frequency for sol in sols]./w0
  kzs = [para(sol.wavenumber) for sol in sols]./k0
  k⊥s = [perp(sol.wavenumber) for sol in sols]./k0
  xk⊥s = sort(unique(k⊥s))
  ykzs = sort(unique(kzs))

  function make2d(z1d)
    z2d = Array{Union{Float64, Missing}, 2}(zeros(Missing, length(ykzs),
                                            length(xk⊥s)))
    for (j, k⊥) in enumerate(xk⊥s), (i, kz) in enumerate(ykzs)
      index = findlast((k⊥ .== k⊥s) .& (kz .== kzs))
      isnothing(index) || (z2d[i, j] = z1d[index])
    end
    return z2d
  end

  ks = [abs(sol.wavenumber) for sol in sols]./k0
  kθs = atan.(k⊥s, kzs)
  extremaangles = collect(extrema(kθs))

  _median(patch) = median(filter(x->!ismissing(x), patch))
  realωssmooth = make2d(real.(ωs))
  try
    realωssmooth = mapwindow(_median, realωssmooth, (5, 5))
    realωssmooth = imfilter(realωssmooth, Kernel.gaussian(3))
  catch
    @warn "Smoothing failed"
  end

  realspline = nothing
  imagspline = nothing
  try
    smoothing = length(ωs) * 1e-4
    realspline = Dierckx.Spline2D(xk⊥s, ykzs, realωssmooth'; kx=4, ky=4, s=smoothing)
    imagspline = Dierckx.Spline2D(k⊥s, kzs, imag.(ωs); kx=4, ky=4, s=smoothing)
  catch err
    @warn "Caught $err. Continuing."
  end

  function plotangles(;writeangles=true)
    for θdeg ∈ vcat(collect.((30:5:80, 81:99, 100:5:150))...)
      θ = θdeg * π / 180
      xs = sin(θ) .* [0, maximum(ks)]
      ys = cos(θ) .* [0, maximum(ks)]
      linestyle = mod(θdeg, 5) == 0 ? :solid : :dash
      Plots.plot!(xs, ys, linecolor=:grey, linewidth=0.5, linestyle=linestyle)
      writeangles || continue
      if atan(maximum(k⊥s), maximum(kzs)) < θ < atan(maximum(k⊥s), minimum(kzs))
        xi, yi = xs[end], ys[end]
        isapprox(yi, maximum(kzs), rtol=0.01, atol=0.01) && (yi += 0.075)
        isapprox(xi, maximum(k⊥s), rtol=0.01, atol=0.01) && (xi += 0.1)
        isapprox(xi, minimum(k⊥s), rtol=0.01, atol=0.01) && (xi -= 0.2)
        Plots.annotate!([(xi, yi, text("\$ $(θdeg)^{\\circ}\$", fontsize, :black))])
      end
    end
  end
  function plotcontours(spline, contourlevels, skipannotation=x->false)
    isnothing(spline) && return nothing
    x, y = sort(unique(k⊥s)), sort(unique(kzs))
    z = evalgrid(spline, x, y)
    for cl ∈ Contour.levels(Contour.contours(x, y, z, contourlevels))
      lvl = try; Int(Contour.level(cl)); catch; Contour.level(cl); end
      for line ∈ Contour.lines(cl)
          xs, ys = Contour.coordinates(line)
          θs = atan.(xs, ys)
          ps = sortperm(θs, rev=true)
          xs, ys, θs = xs[ps], ys[ps], θs[ps]
          mask = minimum(extremaangles) .< θs .< maximum(extremaangles)
          any(mask) || continue
          xs, ys = xs[mask], ys[mask]
          Plots.plot!(xs, ys, color=:grey, linewidth=0.5)
          skipannotation(ys) && continue
          yi, index = findmax(ys)
          xi = xs[index]
          if !(isapprox(xi, minimum(k⊥s), rtol=0.01, atol=0.5) || 
               isapprox(yi, maximum(kzs), rtol=0.01, atol=0.01))
            continue
          end
          isapprox(xi, maximum(k⊥s), rtol=0.1, atol=0.5) && continue
          isapprox(yi, maximum(kzs), rtol=0.1, atol=0.5) && (yi += 0.075)
          isapprox(xi, minimum(k⊥s), rtol=0.1, atol=0.5) && (xi = -0.1)
          Plots.annotate!([(xi, yi, text("\$\\it{$lvl}\$", fontsize-1, :black))])
      end
    end
  end

  msize = 2
  mshape = :square
  function plotter2d(z, xlabel, ylabel, colorgrad,
      climmin=minimum(z[@. !ismissing(z)]), climmax=maximum(z[@. !ismissing(z)]))
    zcolor = make2d(z)
    dx = (xk⊥s[2] - xk⊥s[1]) / (length(xk⊥s) - 1)
    dy = (ykzs[2] - ykzs[1]) / (length(ykzs) - 1)
    h = Plots.heatmap(xk⊥s, ykzs, zcolor, framestyle=:box, c=colorgrad,
      xlims=(minimum(xk⊥s) - dx/2, maximum(xk⊥s) + dx/2),
      ylims=(minimum(ykzs) - dy/2, maximum(ykzs) + dy/2),
      clims=(climmin, climmax), xticks=0:Int(round(maximum(xk⊥s))),
      xlabel=xlabel, ylabel=ylabel)
  end
  xlabel = "\$\\mathrm{Perpendicular\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
  ylabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
  zs = real.(ωs)
  climmax = maximum(zs)
  plotter2d(zs, xlabel, ylabel, Plots.cgrad(), 0.0, climmax)
  Plots.title!(" ")
  plotangles(writeangles=false)
  Plots.plot!(legend=false)
  Plots.savefig(file_extension*"/ICE2D_real_$file_extension.png")

  ω0s = [fastzerobetamagnetoacousticfrequency(Va, sol.wavenumber, Ω1) for
    sol in sols] / w0
  zs = real.(ωs) ./ ω0s
  climmin = minimum(zs)
  climmax = maximum(zs)
  plotter2d(zs, xlabel, ylabel, Plots.cgrad(), climmin, climmax)
  Plots.title!(" ")
  plotangles(writeangles=false)
  Plots.plot!(legend=false)
  Plots.savefig(file_extension*"/ICE2D_real_div_guess_$file_extension.png")

  zs = iseven.(Int64.(floor.(real.(ωs))))
  climmax = maximum(zs)
  plotter2d(zs, xlabel, ylabel, Plots.cgrad(), 0.0, climmax)
  Plots.title!(" ")
  plotangles(writeangles=false)
  plotcontours(realspline, collect(1:50), y -> y[end] < 0)
  Plots.plot!(legend=false)
  Plots.savefig(file_extension*"/ICE2D_evenfloorreal_real_$file_extension.png")

  zs = imag.(ωs)
  climmax = maximum(zs)
  colorgrad = Plots.cgrad([:cyan, :black, :darkred, :red, :orange, :yellow])
  plotter2d(zs, xlabel, ylabel, colorgrad, -climmax / 4, climmax)
  Plots.title!(" ")
  Plots.plot!(legend=false)
  plotcontours(realspline, collect(1:50), y -> y[end] < 0)
  plotangles(writeangles=false)
  Plots.savefig(file_extension*"/ICE2D_imag_$file_extension.png")

  colorgrad = Plots.cgrad()

  xlabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
  ylabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"

  imaglolim = 1e-5

  mask = shuffle(findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= 12)))
  @warn "Scatter plots rendering with $(length(mask)) points."
  perm = sortperm(imag.(ωs[mask]))
  h0 = Plots.scatter(real.(ωs[mask][perm]), kzs[mask][perm],
     zcolor=imag.(ωs[mask][perm]), framestyle=:box, lims=:round,
    markersize=msize+1, markerstrokewidth=0, 
    markershape=:circle, lw=0, msc=:auto,
    c=colorgrad, xticks=(0:12), yticks=unique(Int.(round.(ykzs))),
    xlabel=xlabel, ylabel=ylabel, legend=:topleft)
  Plots.plot!(legend=false)
  Plots.savefig(file_extension*"/ICE2D_KF12_$file_extension.png")


  ylabel = "\$\\mathrm{Growth\\ Rate} \\ [\\Omega_{i}]\$"
  mask = shuffle(findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= 12)))
  h1 = Plots.scatter(real.(ωs[mask]), imag.(ωs[mask]),
    zcolor=kzs[mask], framestyle=:box, lims=:round,
    markersize=msize+1, markerstrokewidth=0, markershape=:circle,
    c=colorgrad, xticks=(0:12), lw=0, msc=:auto,
    xlabel=xlabel, ylabel=ylabel, legend=:topleft)
  Plots.plot!(legend=false)
  Plots.savefig(file_extension*"/ICE2D_F12_$file_extension.png")

  colorgrad1 = Plots.cgrad([:cyan, :red, :blue, :orange, :green,
                            :black, :yellow])
  mask = shuffle(findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= 12)))
  h2 = Plots.scatter(real.(ωs[mask]), imag.(ωs[mask]),
    zcolor=(real.(ωs[mask]) .- vαz/Va .* kzs[mask]), framestyle=:box, lims=:round,
    markersize=msize+1, markerstrokewidth=0, markershape=:circle,
    c=colorgrad1, clims=(0, 13), xticks=(0:12), lw=0, msc=:auto,
    xlabel=xlabel, ylabel=ylabel, legend=:topleft)
  Plots.plot!(legend=false)
  Plots.savefig(file_extension*"/ICE2D_F12_Doppler_$file_extension.png")

  xlabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
  ylabel = "\$\\mathrm{Propagation\\ Angle} \\ [^{\\circ}]\$"

  colorgrad = Plots.cgrad([:cyan, :black, :darkred, :red, :orange, :yellow])
  mask = findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= 12))
  h4 = Plots.scatter(real.(ωs[mask]), kθs[mask] .* 180 / π,
    zcolor=imag.(ωs[mask]), lims=:round,
    markersize=msize, markerstrokewidth=0, markershape=mshape, framestyle=:box,
    c=Plots.cgrad([:black, :darkred, :red, :orange, :yellow]),
    clims=(0, maximum(imag.(ωs[mask]))), lw=0, msc=:auto,
    yticks=(0:10:180), xticks=(0:12), xlabel=xlabel, ylabel=ylabel)
  Plots.plot!(legend=false)
  Plots.savefig(file_extension*"/ICE2D_TF12_$file_extension.png")

  function relative(p, rx, ry)
    xlims = Plots.xlims(p)
    ylims = Plots.ylims(p)
    return xlims[1] + rx * (xlims[2]-xlims[1]), ylims[1] + ry * (ylims[2] - ylims[1])
   end
  Plots.xlabel!(h1, "")
  Plots.xticks!(h1, 0:-1)
  if pitchanglecosine == -0.646
    xy_data = readdlm("data.csv", ',', Float64)
    x_data = xy_data[:, 1]
    y_data = xy_data[:, 2]
    y_data .-= minimum(y_data)
    y_data ./= (maximum(y_data) - minimum(y_data))
    y_data .*= maximum(imag.(ωs[mask]))
    x_data .*= 1e6 / (w0/2π)
    Plots.plot!(h1, x_data, y_data, color=:black)
  end
  Plots.annotate!(h1, [(relative(h1, 0.02, 0.95)..., text("(a)", fontsize, :black))])
  Plots.annotate!(h0, [(relative(h0, 0.02, 0.95)..., text("(b)", fontsize, :black))])
  Plots.plot(h1, h0, link=:x, layout=@layout [a; b])
  Plots.savefig(file_extension*"/ICE2D_Combo_$file_extension.png")
end

if true
  @time plasmasols = findsolutions(Smmr)
  plasmasols = selectlargeestgrowthrate(plasmasols)
  mkdir(name_extension)
  @show length(plasmasols)
  @time plotit(plasmasols)
  @save name_extension*"/solutions2D_$name_extension.jld" filecontents plasmasols w0 k0
  rmprocs(nprocsadded)
else
  cd(name_extension)
  rmprocs(nprocsadded)
  @load name_extension*"/solutions2D_$name_extension.jld" filecontents solutions w0 k0
  @time plotit(solutions)
end

println("Ending at ", now())


