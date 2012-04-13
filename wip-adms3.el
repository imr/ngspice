(mkdir "../build-wip-adms3")

(compile "./autogen.sh")

(defun compile% (command)
  (let ((default-directory (expand-file-name "../build-wip-adms3/"))
        (process-environment (list*
                              (substitute-env-vars "PATH=$PATH:$(readlink -f ../../adms/adms-2.3.0-1500)")
                              process-environment)))
    (compile command)))


(compile% "../ngspice/configure --enable-adms3")
(compile% "LC_ALL=C make -j3")


;;; das admsXml ist hier vom autoconf ? bereits no ist nicht ...
;;;   in den makefile sind abs path drin

(compile "LC_ALL=C make -C src/spicelib/devices/adms/ekv/adms3va top_srcdir=$(pwd) top_builddir=$(readlink -f ../build-adms3) ADMS=$(readlink -f ../../adms/adms-2.3.0-1500/admsXml)")
(compile "LC_ALL=C make -C src/spicelib/devices/adms/hicum0/adms3va top_srcdir=$(pwd) top_builddir=$(readlink -f ../build-adms3) ADMS=$(readlink -f ../../adms/adms-2.3.0-1500/admsXml)")
(compile "LC_ALL=C make -C src/spicelib/devices/adms/myvares/adms3va top_srcdir=$(pwd) top_builddir=$(readlink -f ../build-adms3) ADMS=$(readlink -f ../../adms/adms-2.3.0-1500/admsXml)")


(compile "LC_ALL=C make -C src/spicelib/devices/adms/ekv/adms3va top_srcdir=$(pwd) clean")
(compile "LC_ALL=C make -C src/spicelib/devices/adms/hicum0/adms3va top_srcdir=$(pwd) clean")
(compile "LC_ALL=C make -C src/spicelib/devices/adms/myvares/adms3va top_srcdir=$(pwd) clean")


;;; jetzt passts, ergebnis wie bei laurent ? check ...
;;; execute

(compile "LD_LIBRARY_PATH=src/spicelib/devices/adms/hicum0/adms3va \
   SPICE_SCRIPTS=. \
   ../build-wip-adms3/src/ngspice -b hic0.cir")


(compile% "LC_ALL=C make -j3")
(compile "LD_LIBRARY_PATH=src/spicelib/devices/adms/hicum0/adms3va:src/spicelib/devices/adms/myvares/adms3va \
   ../build-wip-adms3/src/ngspice -p < hic0+.cir")



;;; expected:
Model issue on line 0 : .model x1:hic0_full hicum0 is=1.3525e-18 vef=8.0 iqf=3.0 ...
unrecognized parameter (hicum0) - ignored
;;; expected
0       2.000000e-001   2.996258e-012
1       7.000000e-001   -8.84510e-007

