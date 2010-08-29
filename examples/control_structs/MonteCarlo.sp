Perform Monte Carlo simulation in ngspice
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
*
.control
  let mc_runs = 100
  let run = 1
  set curplot = new       $ create a new plot
  set scratch = $curplot  $ store its name to 'scratch'
*
  define unif(nom, var) (nom + (nom*var) * sunif(0))
  define aunif(nom, avar) (nom + avar * sunif(0))
  define gauss(nom, var, sig) (nom + (nom*var)/sig * sgauss(0))
  define agauss(nom, avar, sig) (nom + avar/sig * sgauss(0))
*
  dowhile run <= mc_runs
*    alter c1 = unif(1e-09, 0.1)
*    alter l1 = aunif(10e-06, 2e-06)
*    alter c2 = aunif(1e-09, 100e-12)
*    alter l2 = unif(10e-06, 0.2)
*    alter l3 = aunif(40e-06, 8e-06)
*    alter c3 = unif(250e-12, 0.15)
    alter c1 = gauss(1e-09, 0.1, 3)
    alter l1 = agauss(10e-06, 2e-06, 3)
    alter c2 = agauss(1e-09, 100e-12, 3)
    alter l2 = gauss(10e-06, 0.2, 3)
    alter l3 = agauss(40e-06, 8e-06, 3)
    alter c3 = gauss(250e-12, 0.15, 3)
    ac oct 100 250K 10Meg
    set run ="$&run"     $ create a variable from the vector
    set dt = $curplot    $ store the current plot to dt
    setplot $scratch     $ make 'scratch' the active plot
    let vout{$run}={$dt}.v(out) $ store the output vector to plot 'scratch'
    setplot $dt          $ go back to the previous plot
    let run = run + 1 
  end
  plot db({$scratch}.all)
.endc

.end
