*ng_script
* Example script for Monte Carlo with commercial HSPICE-compatible libraries
* The circuit in mc_ring_circ.net is a 25-stage inverter ring oscillator.
* Add your library to mc_ring_circ.net and choose transistors accordingly.
* Add the source file and the library path.
* A simple BSIM3 inverter R.O. serves as an MC example wtihout need for a library.
.control
begin
  let mc_runs = 30                $ number of runs for monte carlo
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
  unlet mc_runsp

  set mc_runs = $&mc_runs         $ create a variable from the vector
  let seeds = mc_runs + 2
  setseed $&seeds
  unlet seeds

  echo source the input file
* Path of your circuit file and library file here
* Will be added to the already existing sourcepath
  setcs sourcepath = ( $inputdir $sourcepath ./ngspice/examples/Monte_Carlo )
* source with file name of your circuit file
  source mc_ring_circ.net

  save buf                        $ we just need buf, save memory by more than 10x

* Output path (directory has already to be there)
*  set outputpath = 'D:\Spice_general\ngspice\examples\Monte_Carlo\out'
* If your current directory is the 'ngspice' directory
*  set outputpath = './examples/Monte_Carlo/out' $ LINUX alternative
* run the simulation loop

* We have to figure out what to do if a single simulation will not converge.
* There is now the variable sim_status, that is 0 if simulation ended regularly,
* and 1 if the simulation has been aborted with error message '...simulation(s) aborted'.
* Then we skip the rest of the run and continue with a new run.

  dowhile run <= mc_runs

    set run = $&run               $ create a variable from the vector

    * run=0 simulates with nominal parameters
    if run > 0
      echo
      echo * * * * * *
      echo Source the circuit again internally for run no. $run
      echo * * * * * *
      setseed $run
      mc_source  $ re-source the input file
    else
      echo run no. $run
    end
    echo simulation run no. $run of $mc_runs
    tran 100p 1000n 0
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

* select stop and step so that number of data points after linearization is not too
* close to 8192, which would yield varying number of line length and thus scale for fft.
*
    set dt0 = $curplot
    * save the linearized data for having equal time scales for all runs
    linearize buf                 $ linearize only buf, no other vectors needed
    set dt1 = $curplot             $ store the current plot to dt (tran i+1)
    setplot $plot_out             $ make 'plt_out' the active plot
    * firstly save the time scale once to become the default scale
    if run=0
       let time={$dt1}.time
    end
    let vout{$run}={$dt1}.buf     $ store the output vector to plot 'plot_out'
    setplot $dt1                  $ go back to the previous plot (tran i+1)
    fft buf $ run fft on vector buf
    let buf2=db(mag(buf))
    * find the frequency where buf has its maximum of the fft signal
    meas sp fft_max MAX_AT buf2 from=0.05G to=0.7G
    * find the frequency where buf is -40dB at rising fft signal
    meas sp fft_40 WHEN buf2=-40 RISE=1 from=0.05G to=0.7G
    * store the fft vector
    set dt2 = $curplot            $ store the current plot to dt (spec i)
    setplot $plot_fft             $ make 'plot_fft' the active plot
    if run=0
       let frequency={$dt2}.frequency
    end
    let fft{$run}={$dt2}.buf      $ store the output vector to plot 'plot_fft'
    settype decibel fft{$run}
    * store the measured value
    setplot $max_fft              $ make 'max_fft' the active plot
    let maxffts[{$run}]={$dt2}.fft_max
    let halfffts[{$run}]={$dt2}.fft_40
    destroy $dt0  $dt1  $dt2      $ save memory, we don't need this plot (spec) any more

    label next
    remcirc
    let run = run + 1
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
  set nolegend
  if $?sharedmode or $?win_console
    gnuplot xnp_pl1 {$plot_out}.vout0 ylabel vout0   $ just plot the tran output with nominal parameters
  else
    plot {$plot_out}.vout0  ylabel vout0    $ just plot the tran output with nominal parameters
  end
  setplot $plot_fft
  if $?sharedmode or $?win_console
    gnuplot xnp_pl2 db(mag(ally)) ylabel 'output voltage versus frequency' xlimit 0 1G ylimit -80 10
  else
    plot db(mag(ally)) xlimit 0 1G ylimit -80 10 ylabel 'output voltage versus frequency'
  end
*
* create a histogram from vector maxffts
  setplot $max_fft                $ make 'max_fft' the active plot
  set startfreq=50MEG
  set bin_size=1MEG
  set bin_count=100
  compose osc_frequ start=$startfreq step=$bin_size lin=$bin_count $ requires variables as parameters
  settype frequency osc_frequ
  let bin_count=$bin_count        $ create a vector from the variable
  let yvec=unitvec(bin_count)     $ requires vector as parameter
  let startfreq=$startfreq
  let bin_size=$bin_size
  * put data into the correct bins
  let run = 0
  dowhile run < mc_runs
    set run = $&run              $ create a variable from the vector
    let val = maxffts[{$run}]
    let part = 0
    * Check if val fits into a bin. If yes, raise bin by 1
    dowhile part < bin_count
      if ((val < (startfreq + (part+1)*bin_size)) & (val >= (startfreq + part*bin_size)))
        let yvec[part] = yvec[part] + 1
                break
      end
      let part = part + 1
    end
    let run = run + 1
  end
  * plot the histogram
  let count = yvec - 1             $ subtract 1 because we started with unitvec containing ones
  if $?sharedmode or $?win_console
    gnuplot np_pl3 count vs osc_frequ combplot ylabel 'counts per bin'
  else
    set xbrushwidth=5
    plot count vs osc_frequ combplot ylabel 'counts per bin'
  end
* calculate jitter
  let diff40 = (vecmax(halfffts) - vecmin(halfffts))*1e-6
  echo
  echo Max. jitter is "$&diff40" MHz
end
  rusage
*  quit
end
