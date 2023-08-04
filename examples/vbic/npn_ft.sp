VBIC ft Test

.include Infineon_VBIC.lib

vce 1 0 dc 3.0
vgain 1 c dc 0.0
f 0 2 vgain -1000
l 2 b 1g
c 2 0 1g
ib 0 b dc 0.0 ac 1.0
ic 0 c 0.01
xq1 c b 0 0 BFP780

.control
let ft_runs = 13
let run = 0
set curplot=new          $ create a new plot
set curplotname=ft_plot
set curplottitle="Infineon BFP780 ft = f(Ic)"
set scratch=$curplot     $ store its name to 'scratch'
setplot $scratch         $ make 'scratch' the active plot 
let ft=unitvec(ft_runs)  $ create a vector in plot 'scratch' to store ft data 
let ic=unitvec(ft_runs)  $ create a vector in plot 'scratch' to store ic data 
foreach myic 2m 5m 9m 15m 22m 30m 50m 75m 90m 100m 110m 130m 180m
 alter ic = $myic
 ac dec 100 100k 50g
*plot vdb(vgain#branch)
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

.end
