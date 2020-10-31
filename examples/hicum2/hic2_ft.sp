HICUM2v2.40 Test ft=f(Ic) Vce=1V

vce 1 0 dc 1.0
vgain 1 c dc 0.0
f 0 2 vgain -2
l 2 b 1g
c 2 0 1g
ib 0 b dc 0.0 ac 1.0
ic 0 c 0.001
Q1 C B 0 hicumL2V2p40

.control
let run = 0
let ft_runs = 9
set curplot=new          $ create a new plot
set curplotname=ft_plot
set curplottitle="HICUM2v2.40 ft = f(Ic)"
set scratch=$curplot     $ store its name to 'scratch'
setplot $scratch         $ make 'scratch' the active plot 
let ft=unitvec(ft_runs)  $ create a vector in plot 'scratch' to store ft data 
let ic=unitvec(ft_runs)  $ create a vector in plot 'scratch' to store ic data 
foreach myic 1e-03 3e-03 6e-03 9e-03 14e-03 21e-03 27e-03 33e-3 40e-03
 alter ic = $myic
 op
 print all
 ac dec 100 1Meg 800g
 meas ac freq_at when vdb(vgain#branch)=0
 set run ="$&run"            $ create a variable from the vector
 set dt = $curplot           $ store the current plot to dt
 setplot $scratch            $ make 'scratch' the active plot
 let ic[run] = $myic         $ store ic to vector ft in plot 'scratch'
 let ft[run] = {$dt}.freq_at $ store ft to vector ft in plot 'scratch'
 setplot $dt                 $ go back to the previous plot
 let run = run + 1
end
setplot unknown1
plot ft vs ic xlog
.endc

.include model-card-examples.lib

.end
