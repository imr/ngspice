// (compile "git ls-files -- 'src/spicelib/devices/sw/*.c' 'src/spicelib/devices/csw/*.c' | xargs -P8 -n1 spatch --smpl-spacing --sp-file rewrite-sw.cocci --in-place && git commit -am 'rewrite-sw, coccinelle semantic patch #1'")
// (compile "git ls-files -- 'src/spicelib/devices/sw/*.c' 'src/spicelib/devices/csw/*.c' | xargs -P8 -n1 spatch --smpl-spacing --sp-file rewrite-sw.cocci")


@r1@
expression e;
symbol ckt, CKTrhsOld;
@@
- *(ckt->CKTrhsOld + e)
+ ckt->CKTrhsOld[e]


@r2@
expression e;
constant int c;
identifier here;
identifier Wstate = {SWstate, CSWstate};
symbol ckt, CKTstates, CKTstate0, CKTstate1;
@@
(
- *(ckt->CKTstates[c] + (here->Wstate))
+ ckt->CKTstates[c][here->Wstate + 0]
|
- *(ckt->CKTstate0 + (here->Wstate))
+ ckt->CKTstates[0][here->Wstate + 0]
|
- *(ckt->CKTstate1 + (here->Wstate))
+ ckt->CKTstates[1][here->Wstate + 0]
|
- *(ckt->CKTstates[c] + here->Wstate + e)
+ ckt->CKTstates[c][here->Wstate + e]
|
- *(ckt->CKTstate0 + (here->Wstate + e))
+ ckt->CKTstates[0][here->Wstate + e]
|
- *(ckt->CKTstate1 + (here->Wstate + e))
+ ckt->CKTstates[1][here->Wstate + e]
)


@r3@
identifier i;
@@
- return(i);
+ return i;


@r4@
statement S;
identifier x;
@@
  for (...;
- x != NULL;
+ x;
  ...) S
