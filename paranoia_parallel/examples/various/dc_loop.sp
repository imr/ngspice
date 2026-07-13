testing loops
*variables are global
*vector reside only in the plot where they where created

R1 r2 0 r = {5k + 50*TEMPER}
V1 r2 0 1

.control
*create a new plot as our base plot
setplot new
set curplottitle = "crossplot"
set plotname=$curplot

* generate vector with all (here 30) elements
let result=vector(30)
* reshape vector to format 5 x 6
reshape result [5][6]
* vector to store temperature
let tvect=vector(5)

*index for storing in vectors tvect and result
let index = 0

foreach var -40 -20 0 20 40
  set temp = $var
  dc v1 0 5 1
  *store name of the actual dc plot
  set dcplotname = $curplot
  * back to the base plot
  setplot $plotname
  let result[index] = {$dcplotname}.v1#branch
  let tvect[index] = $var
  settype current result
  let index = index + 1
  destroy $dcplotname
end

settype temp-sweep tvect
setscale tvect

transpose result

plot result
write dc_loop.out result[0] result[1] result[2] result[3] result[4] result[5]

.endc

.end
