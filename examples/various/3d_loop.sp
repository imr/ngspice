testing loops
*variables are global
*vector reside only in the plot where they where created

.param rr = 10k

R1 r2 0 r = {rr + 40*TEMPER}
V1 r2 0 1

.control
*create a new plot as our base plot
setplot new
set curplottitle = "crossplot"
set plotname=$curplot

let aa = 5
let bb = 3
let cc = 6
set aa="$&aa"
set bb="$&bb"
set cc="$&cc"

* generate vector with all (here 90) elements
let result=vector(90)
settype current result
* reshape vector to format 5 x 3 x 6
*reshape result [5][3][6]
reshape result [$aa][$bb][$cc]

* vector to store temperature
let tvect=vector(5)
* vector to store voltage
let vvect=vector(6)
* vector to store parameter values
let pvect=vector(3)

*index for storing in vectors tvect and result

let indexp = 0

foreach pvar 9.5k 10k 10.5k
  let indexv = 0
  alterparam rr = $pvar
  let pvect[indexp] = $pvar
  mc_source
  foreach var -40 -20 0 20 40
    set temp = $var
    dc v1 0 5 1
    *store name of the actual dc plot
    set dcplotname = $curplot
    * back to the base plot
    setplot $plotname
    let result[indexv][indexp] = {$dcplotname}.v1#branch
    let tvect[indexv] = $var
    if indexv = 0
      let vvect = {$dcplotname}.r2
    end
    let indexv = indexv + 1
*    destroy $dcplotname
  end
  let indexp = indexp + 1
  remcirc
end

settype voltage vvect
setscale vvect

let indexplot = 0
while indexplot < indexp
*plot result[0][indexplot] result[1][indexplot] result[2][indexplot] result[3][indexplot] result[4][indexplot]
let indexplot = indexplot + 1
end


plot
+result[0][0] result[1][0] result[2][0] result[3][0] result[4][0]
+result[0][1] result[1][1] result[2][1] result[3][1] result[4][1]
+result[0][2] result[1][2] result[2][2] result[3][2] result[4][2]

write 3d_loop_i_vs_v.out
+result[0][0] result[1][0] result[2][0] result[3][0] result[4][0]
+result[0][1] result[1][1] result[2][1] result[3][1] result[4][1]
+result[0][2] result[1][2] result[2][2] result[3][2] result[4][2]

*transpoe a 3D vector
let aai = 0
let bbi = 0
let cci = 0
let result1 = vector(90)
settype current result1
* reshape vector to format 3 x 6 x 5
reshape result1 [$bb][$cc][$aa]

* shift from vector format 5 x 3 x 6 to 3 x 6 x 5
*echo test output > resultout.txt
while aai < aa
  let bbi = 0
  while bbi < bb
    let cci = 0
    while cci < cc
      let result1[bbi][cci][aai] = result[aai][bbi][cci]
*      print bbi cci aai >> resultout.txt
*      print result1[bbi][cci][aai] >> resultout.txt
      let cci = cci + 1
    end
    let bbi = bbi + 1
  end
  let aai = aai + 1
end

settype temp-sweep tvect
setscale tvect

* current through v1 versus temperature
plot
+result1[0][0] result1[1][0] result1[2][0]
+result1[0][1] result1[1][1] result1[2][1]
+result1[0][2] result1[1][2] result1[2][2]
+result1[0][3] result1[1][3] result1[2][3]
+result1[0][4] result1[1][4] result1[2][4]
+result1[0][5] result1[1][5] result1[2][5]

write 3d_loop_i_vs_t.out
+result1[0][0] result1[1][0] result1[2][0]
+result1[0][1] result1[1][1] result1[2][1]
+result1[0][2] result1[1][2] result1[2][2]
+result1[0][3] result1[1][3] result1[2][3]
+result1[0][4] result1[1][4] result1[2][4]
+result1[0][5] result1[1][5] result1[2][5]

*plot result1

*transpoe a 3D vector
let aai = 0
let bbi = 0
let cci = 0
let result2 = vector(90)
settype current result2
* reshape vector to format 6 x 5 x 3
reshape result2 [$cc][$aa][$bb]

* shift from vector format 3 x 6 x 5 to 6 x 5 x 3
*echo test output > resultout.txt
while aai < aa
  let bbi = 0
  while bbi < bb
    let cci = 0
    while cci < cc
      let result2[cci][aai][bbi] = result1[bbi][cci][aai]
*      print cci aai bbi >> resultout.txt
*      print result2[cci][aai][bbi] >> resultout.txt
      let cci = cci + 1
    end
    let bbi = bbi + 1
  end
  let aai = aai + 1
end

settype impedance pvect
setscale pvect

* current through v1 versus parameter rr
plot
+result2[0][0] result2[1][0] result2[2][0] result2[3][0] result2[4][0] result2[5][0]
+result2[0][1] result2[1][1] result2[2][1] result2[3][1] result2[4][1] result2[5][1]
+result2[0][2] result2[1][2] result2[2][2] result2[3][2] result2[4][2] result2[5][2]
+result2[0][3] result2[1][3] result2[2][3] result2[3][3] result2[4][3] result2[5][3]
+result2[0][4] result2[1][4] result2[2][4] result2[3][4] result2[4][4] result2[5][4]

write 3d_loop_i_vs_para.out
+result2[0][0] result2[1][0] result2[2][0] result2[3][0] result2[4][0] result2[5][0]
+result2[0][1] result2[1][1] result2[2][1] result2[3][1] result2[4][1] result2[5][1]
+result2[0][2] result2[1][2] result2[2][2] result2[3][2] result2[4][2] result2[5][2]
+result2[0][3] result2[1][3] result2[2][3] result2[3][3] result2[4][3] result2[5][3]
+result2[0][4] result2[1][4] result2[2][4] result2[3][4] result2[4][4] result2[5][4]

.endc

.end
