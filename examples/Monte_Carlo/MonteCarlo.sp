* Effecting a Monte Carlo calculation in ngspice
V1 N001 0 AC 1 DC 0
R1 N002 N001 141
*
C1 OUT 0 1e-09
L1 OUT 0 10e-06
C2 N002 0 1e-09
L2 N002 0 10e-06
L3 N003 N002 40e-06
C3 OUT N003 250e-12
*
R2 0 OUT 141
.control
  let mc_runs = 5
  let run = 0
  set curplot=new          $ create a new plot
  set scratch=$curplot     $ store its name to 'scratch'
  setplot $scratch         $ make 'scratch' the active plot
  let bwh=unitvec(mc_runs) $ create a vector in plot 'scratch' to store bandwidth data

* define distributions for random numbers:
* unif: uniform distribution, deviation relativ to nominal value
* aunif: uniform distribution, deviation absolut
* gauss: Gaussian distribution, deviation relativ to nominal value
* agauss: Gaussian distribution, deviation absolut
* limit: if unif. distributed value >=0 then add +avar to nom, else -avar
  define unif(nom, rvar) (nom + (nom*rvar) * sunif(0))
  define aunif(nom, avar) (nom + avar * sunif(0))
  define gauss(nom, rvar, sig) (nom + (nom*rvar)/sig * sgauss(0))
  define agauss(nom, avar, sig) (nom + avar/sig * sgauss(0))
*  define limit(nom, avar) (nom + ((sgauss(0) ge 0) ? avar : -avar))
  define limit(nom, avar) (nom + ((sgauss(0) >= 0) ? avar : -avar))
*
*
  dowhile run < mc_runs    $ loop starts here
*
*    alter c1 = unif(1e-09, 0.1)
*    alter c1 = aunif(1e-09, 100e-12)
*    alter c1 = gauss(1e-09, 0.1, 3)
*    alter c1 = agauss(1e-09, 100e-12, 3)
*
    alter c1 = unif(1e-09, 0.1)
    alter l1 = unif(10e-06, 0.1)
    alter c2 = unif(1e-09, 0.1)
    alter l2 = unif(10e-06, 0.1)
    alter l3 = unif(40e-06, 0.1)
    alter c3 = limit(250e-12, 25e-12)
*
    ac oct 100 250K 10Meg
*
* measure bandwidth at -10 dB
    meas ac bw trig vdb(out) val=-10 rise=1 targ vdb(out) val=-10 fall=1
*
    set run = $&run             $ create a variable from the vector
    set dt = $curplot           $ store the current plot to dt
    setplot $scratch            $ make 'scratch' the active plot
    let vout{$run}={$dt}.v(out) $ store the output vector to plot 'scratch'
    let bwh[run]={$dt}.bw       $ store bw to vector bwh in plot 'scratch'
    setplot $dt                 $ go back to the previous plot
    let run = run + 1
  end    $ loop ends here
*
  plot db({$scratch}.allv)
  echo
  print {$scratch}.bwh
.endc

.end
