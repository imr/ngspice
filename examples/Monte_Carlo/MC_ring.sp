Perform Monte Carlo simulation in ngspice
* 25 stage Ring-Osc. BSIM3 with statistical variation of various model parameters
* cd into ngspice/examples/Monte_Carlo
* start in interactive mode 'ngspice MC_ring.sp' with several plots for output
* or start in batch mode, controlled by .control section (Control mode)
* with 'ngspice -b -r MC_ring.raw -o MC_ring.log MC_ring.sp'.

vin in out dc 0.5 pulse 0.5 0 0.1n 5n 1 1 1
vdd dd 0 dc 3.3
vss ss 0 dc 0
ve  sub  0 dc 0
vpe well 0 dc 3.3

.subckt inv1 dd ss sub well in out
mn1  out in  ss  sub  n1  w=2u l=0.35u as=3p ad=3p ps=4u pd=4u
mp1  out in  dd  well p1  w=4u l=0.35u as=7p ad=7p ps=6u pd=6u
.ends inv1

.subckt inv5 dd ss sub well in out
xinv1 dd ss sub well in 1 inv1
xinv2 dd ss sub well 1  2 inv1
xinv3 dd ss sub well 2  3 inv1
xinv4 dd ss sub well 3  4 inv1
xinv5 dd ss sub well 4 out inv1
.ends inv5

xinv1 dd ss sub well in out5 inv5
xinv2 dd ss sub well out5 out10 inv5
xinv3 dd ss sub well out10 out15 inv5
xinv4 dd ss sub well out15 out20 inv5
xinv5 dd ss sub well out20 out inv5
xinv11 dd 0 sub well out buf inv1
cout  buf ss 0.2pF
*
.options noacct
.control
  save buf                        $ we just need buf, save memory by more than 10x
  let mc_runs = 30             $ number of runs for monte carlo
  let run = 0                     $ number of actual run
  set curplot = new               $ create a new plot
  set curplottitle = "Transient outputs"
  set plot_out = $curplot         $ store its name to 'plot_out'
  set curplot = new               $ create a new plot
  set curplottitle = "FFT outputs"
  set plot_fft = $curplot         $ store its name to 'plot_fft'
  set curplot = new               $ create a new plot
  set curplottitle = "Oscillation frequency"
  set max_fft = $curplot          $ store its name to 'max_fft'
  let mc_runsp = mc_runs + 1
  let maxffts = unitvec(mc_runsp) $ vector for storing max measure results
  let halfffts = unitvec(mc_runsp)$ vector for storing measure results at -40dB rising
*
* define distributions for random numbers:
* unif: uniform distribution, deviation relativ to nominal value
* aunif: uniform distribution, deviation absolut
* gauss: Gaussian distribution, deviation relativ to nominal value
* agauss: Gaussian distribution, deviation absolut
  define unif(nom, var) (nom + (nom*var) * sunif(0))
  define aunif(nom, avar) (nom + avar * sunif(0))
  define gauss(nom, var, sig) (nom + (nom*var)/sig * sgauss(0))
  define agauss(nom, avar, sig) (nom + avar/sig * sgauss(0))
*
* We want to vary the model parameters vth0, u0, tox, lint, and wint
* of the BSIM3 model for the NMOS and PMOS transistors.
* We may obtain the nominal values (nom) by manually extracting them from
* the parameter set. Here we get them automatically and store them into
* vectors. This has the advantage that you may change the parameter set
* without having to look up the values again.
  let n1vth0=@n1[vth0]
  let n1u0=@n1[u0]
  let n1tox=@n1[tox]
  let n1lint=@n1[lint]
  let n1wint=@n1[wint]
  let p1vth0=@p1[vth0]
  let p1u0=@p1[u0]
  let p1tox=@p1[tox]
  let p1lint=@p1[lint]
  let p1wint=@p1[wint]

*
* run the simulation loop
  dowhile run <= mc_runs
    * run=0 simulates with nominal parameters
    if run > 0
      setplot $max_fft
      altermod @n1[vth0] = gauss(n1vth0, 0.1, 3)
      altermod @n1[u0] = gauss(n1u0, 0.05, 3)
      altermod @n1[tox] = gauss(n1tox, 0.1, 3)
      altermod @n1[lint] = gauss(n1lint, 0.1, 3)
      altermod @n1[wint] = gauss(n1wint, 0.1, 3)
      altermod @p1[vth0] = gauss(p1vth0, 0.1, 3)
      altermod @p1[u0] = gauss(p1u0, 0.1, 3)
      altermod @p1[tox] = gauss(p1tox, 0.1, 3 )
      altermod @p1[lint] = gauss(p1lint, 0.1, 3)
      altermod @p1[wint] = gauss(p1wint, 0.1, 3)
    end
    tran 15p 100n 0
* select stop and step so that number of data points after linearization is not too
* close to 8192, which would yield varying number of line length and thus scale for fft.
*
* We have to figure out what to do if a single simulation will not converge.
* There is the variable 'sim_status' which is set to 1 if the simulation
* fails with ’xx simulation(s) aborted’, e.g. because of non-convergence.
* Then we might skip this run and continue with a new run.
*
    echo Simulation status $sim_status
    let simstat = $sim_status
    if simstat = 1
      if run = mc_runs
        echo go to end
      else
        echo go to next run
      end
      destroy $curplot
      goto next
    end

    set run ="$&run"              $ create a variable from the vector
    set mc_runs ="$&mc_runs"      $ create a variable from the vector
    echo simulation run no. $run of $mc_runs
    set dt = $curplot
    * save the linearized data for having equal time scales for all runs
    linearize buf                 $ linearize only buf, no other vectors needed
    destroy $dt                   $ delete the tran i plot
    set dt = $curplot             $ store the current plot to dt (tran i+1)
    setplot $plot_out             $ make 'plt_out' the active plot
    * firstly save the time scale once to become the default scale
    if run=0
       let time={$dt}.time
    end
    let vout{$run}={$dt}.buf      $ store the output vector to plot 'plot_out'
    setplot $dt                   $ go back to the previous plot (tran i+1)
    fft buf $ run fft on vector buf
    destroy $dt                   $ delete the tran i+1 plot
    let buf2=db(mag(buf))
    * find the frequency where buf has its maximum of the fft signal
    meas sp fft_max MAX_AT buf2 from=0.1G to=0.7G
    * find the frequency where buf is -40dB at rising fft signal
    meas sp fft_40 WHEN buf2=-40 RISE=1 from=0.1G to=0.7G
    echo
    echo
    * store the fft vector
    set dt = $curplot             $ store the current plot to dt (spec i)
    setplot $plot_fft             $ make 'plot_fft' the active plot
    if run=0
       let frequency={$dt}.frequency
    end
    let fft{$run}={$dt}.buf       $ store the output vector to plot 'plot_fft'
    * store the measured value
    setplot $max_fft              $ make 'max_fft' the active plot
    let maxffts[{$run}]={$dt}.fft_max
    let halfffts[{$run}]={$dt}.fft_40
    let run = run + 1
    label next
    reset
  end
***** plotting **********************************************************
if $?batchmode
  echo
  echo Plotting not available in batch mode
  echo Write linearized vout0 to vout{$mc_runs} to rawfile $rawfile
  echo
  write $rawfile {$plot_out}.allv
  rusage
  quit
else
  setplot $plot_out
  plot vout0  ylabel 'RO output, original parameters'        $ just plot the tran output with nominal parameters
  setplot $plot_fft
  settype decibel ally
  plot db(mag(ally)) xlimit .1G 1G ylimit -80 10 ylabel 'fft output'
*
* create a histogram from vector maxffts
  setplot $max_fft                $ make 'max_fft' the active plot
  set startfreq=400MEG
  set bin_size=5MEG
  set bin_count=20
  compose xvec start=$startfreq step=$bin_size lin=$bin_count $ requires variables as parameters
  settype frequency xvec
  let bin_count=$bin_count        $ create a vector from the variable
  let yvec=unitvec(bin_count)     $ requires vector as parameter
  let startfreq=$startfreq
  let bin_size=$bin_size
  * put data into the correct bins
  let run = 0
  dowhile run < mc_runs
    set run = $&run               $ create a variable from the vector
    let val = maxffts[{$run}]
    let part = 0
    * Check if val fits into a bin. If yes, raise bin by 1
    dowhile part < bin_count
      if ((val < (startfreq + (part+1)*bin_size)) & (val > (startfreq + part*bin_size)))
        let yvec[part] = yvec[part] + 1
                break
      end
      let part = part + 1
    end
    let run = run + 1
  end

  * plot the histogram
  set plotstyle=combplot
  plot yvec-1 vs xvec   xlabel 'oscillation frequency' ylabel 'bin count'     $ subtract 1 because we started with unitvec containing ones

  * plot simulation series
  set plotstyle=linplot
  let xx = vector(mc_runsp)
  settype frequency maxffts
  plot maxffts vs xx xlabel 'iteration no.' ylabel 'RO frequency'

* calculate jitter
  let diff40 = (vecmax(halfffts) - vecmin(halfffts))*1e-6
  echo
  echo Max. jitter is "$&diff40" MHz
end   
  rusage
.endc
********************************************************************************
.model n1 nmos
+level=8
+version=3.3.0
+tnom=27.0
+nch=2.498e+17  tox=9e-09 xj=1.00000e-07
+lint=9.36e-8 wint=1.47e-7
+vth0=.6322    k1=.756  k2=-3.83e-2  k3=-2.612
+dvt0=2.812  dvt1=0.462  dvt2=-9.17e-2
+nlx=3.52291e-08  w0=1.163e-6
+k3b=2.233
+vsat=86301.58  ua=6.47e-9  ub=4.23e-18  uc=-4.706281e-11
+rdsw=650  u0=388.3203 wr=1
+a0=.3496967 ags=.1    b0=0.546    b1=1
+dwg=-6.0e-09 dwb=-3.56e-09 prwb=-.213
+keta=-3.605872e-02  a1=2.778747e-02  a2=.9
+voff=-6.735529e-02  nfactor=1.139926  cit=1.622527e-04
+cdsc=-2.147181e-05
+cdscb=0  dvt0w=0 dvt1w=0 dvt2w=0
+cdscd=0 prwg=0
+eta0=1.0281729e-02  etab=-5.042203e-03
+dsub=.31871233
+pclm=1.114846  pdiblc1=2.45357e-03  pdiblc2=6.406289e-03
+drout=.31871233  pscbe1=5000000  pscbe2=5e-09 pdiblcb=-.234
+pvag=0 delta=0.01
+wl=0 ww=-1.420242e-09 wwl=0
+wln=0 wwn=.2613948 ll=1.300902e-10
+lw=0 lwl=0 lln=.316394 lwn=0
+kt1=-.3  kt2=-.051
+at=22400
+ute=-1.48
+ua1=3.31e-10  ub1=2.61e-19 uc1=-3.42e-10
+kt1l=0 prt=764.3
+noimod=2
+af=1.075e+00              kf=9.670e-28               ef=1.056e+00
+noia=1.130e+20            noib=7.530e+04             noic=-8.950e-13
**** PMOS ***
.model p1 pmos
+level=8
+version=3.3.0
+tnom=27.0
+nch=3.533024e+17  tox=9e-09 xj=1.00000e-07
+lint=6.23e-8 wint=1.22e-7
+vth0=-.6732829 k1=.8362093  k2=-8.606622e-02  k3=1.82
+dvt0=1.903801  dvt1=.5333922  dvt2=-.1862677
+nlx=1.28e-8  w0=2.1e-6
+k3b=-0.24 prwg=-0.001 prwb=-0.323
+vsat=103503.2  ua=1.39995e-09  ub=1.e-19  uc=-2.73e-11
+rdsw=460  u0=138.7609
+a0=.4716551 ags=0.12
+keta=-1.871516e-03  a1=.3417965  a2=0.83
+voff=-.074182  nfactor=1.54389  cit=-1.015667e-03
+cdsc=8.937517e-04
+cdscb=1.45e-4  cdscd=1.04e-4
+dvt0w=0.232 dvt1w=4.5e6 dvt2w=-0.0023
+eta0=6.024776e-02  etab=-4.64593e-03
+dsub=.23222404
+pclm=.989  pdiblc1=2.07418e-02  pdiblc2=1.33813e-3
+drout=.3222404  pscbe1=118000  pscbe2=1e-09
+pvag=0
+kt1=-0.25  kt2=-0.032 prt=64.5
+at=33000
+ute=-1.5
+ua1=4.312e-9 ub1=6.65e-19  uc1=0
+kt1l=0
+noimod=2
+af=9.970e-01              kf=2.080e-29               ef=1.015e+00
+noia=1.480e+18            noib=3.320e+03             noic=1.770e-13
.end
