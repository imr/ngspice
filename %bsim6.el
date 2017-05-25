(compile "wget http://bsim.berkeley.edu/BSIMBULK/BSIM6.1.1_07302015.tar.gz")

stolen from ...
bash ./testNgspice.sh $(pwd)/../admsXml/admsXml
  Running admsXml for Ngspice
Using: /home/larice/s/ngspice.work/work/ngspice/adms/ADMS/testcases/../admsXml/admsXm

(compile "adms/ADMS/admsXml/admsXml")

(compile "cd src/spicelib/devices/adms/bsim6/admsva && make -if my.mak to")
(compile "cd src/spicelib/devices/adms/bsim6 && ./build.sh")
(ffap "src/spicelib/devices/adms/bsim6")

(compile "cd src/spicelib/devices/adms/hicum0/admsva && make -if my.mak to")
(compile "cd src/spicelib/devices/adms/hicum2/admsva && make -if my.mak to")
(compile "cd src/spicelib/devices/adms/mextram/admsva && make -if my.mak to")
(compile "cd src/spicelib/devices/adms/psp102/admsva && make -if my.mak to")
(compile "cd src/spicelib/devices/adms/ekv/admsva && make -if my.mak to")

;; note only some blessed module names are accepted, see ngspiceVersion.xml
;; here I reuse r2_cmc
(compile "cd src/spicelib/devices/adms/ex-1/admsva && make -if my.mak to")

(compile "cd src/spicelib/devices/adms/bsim6/admsva && ../../../../../../adms/ADMS/admsXml/admsXml -e /home/larice/s/ngspice.work/work/ngspice/adms/ADMS/scripts/vlatovla.xml bsim6.va")
(compile "cd src/spicelib/devices/adms/bsim6/admsva && ../../../../../../adms/ADMS/admsXml/admsXml -e /home/larice/s/ngspice.work/work/ngspice/adms/ADMS/scripts/vlatommlMODULE.htm.xml bsim6.va")

$vt
$param.. in guesstopology
and all the code in bsim..h



../../../../../../ngspice/src/spicelib/devices/adms/bsim6/bsim6guesstopology.c:694:22: error: 'T0' undeclared (first use in this function)
 NDEP_i=(NDEP_i*((1.0+T0)+T1));
                      ^

../../../../../../ngspice/src/spicelib/devices/adms/bsim6/bsim6guesstopology.c:997:3: error: 'GEOMOD' undeclared (first use in this function)
((GEOMOD)==(0))

kommt in /* case */ als GEOMOD statt here->GEOMOD
