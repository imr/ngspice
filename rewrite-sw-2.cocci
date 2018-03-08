// (compile "git ls-files -- 'src/spicelib/devices/sw/*.c' 'src/spicelib/devices/csw/*.c' | xargs -P8 -n1 spatch --smpl-spacing --sp-file rewrite-sw-2.cocci --in-place && git commit -am 'rewrite-sw, coccinelle semantic patch #2'")
// (compile "git ls-files -- 'src/spicelib/devices/sw/*.c' 'src/spicelib/devices/csw/*.c' | xargs -P8 -n1 spatch --smpl-spacing --sp-file rewrite-sw-2.cocci")



@r1@
symbol ckt, CKTstates, CKTstate0, CKTstate1;
@@
(
- ckt->CKTstates[0]
+ ckt->CKTstate0
|
- ckt->CKTstates[1]
+ ckt->CKTstate1
)


@r2@
identifier here;
@@
(
- here->SWstate + 0
+ here->SWswitchstate
|
- here->SWstate + 1
+ here->SWctrlvalue
)


@r3@
identifier here;
@@
(
- here->CSWstate + 0
+ here->CSWswitchstate
|
- here->CSWstate + 1
+ here->CSWctrlvalue
)
