

.subckt dum ss
mn1 ss ss ss ss ss n1 w=4u l=0.15u debug=1 AS=6p AD=6p PS=7u PD=7u
.ends dum

* XOR2
.subckt xnor2 dd ss sub A B out
mn1  T1  A   C1  sub  n1  w=4u  l=0.15u  AS=6p AD=6p PS=7u PD=7u
mn2  C1  B   ss  sub  n1  w=4u  l=0.15u   AS=6p AD=6p PS=7u PD=7u
mn3  out A   C2  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn4  out B   C2  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn5  C2  T1  ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mp1  T1  A   dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp2  T1  B   dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp3  out A   C3  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp4  C3  B   dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp5  out T1  dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
.ends xnor2

.subckt nor2 dd ss sub A B out
mn1  out A   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn2  out B   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mp1  out A   C1  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp2  C1  B   dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
.ends nor2

.subckt nand2 dd ss sub A B out
mn1  out A   C1  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn2  C1  B   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mp1  out A   dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp2  out B   dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
.ends nand2

.subckt nor3 dd ss sub A B C out
mn1  out A   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn2  out B   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn3  out C   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mp1  out A   C1  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp2  C1  B   C2  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp3  C2  C   dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
.ends nor3

.subckt nand3 dd ss sub A B C out
mn1  out A   C1  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn2  C1  B   C2  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn3  C2  C   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mp1  out A   dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp2  out B   dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp3  out C   dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
.ends nand3

.subckt nor4 dd ss sub A B C D out
mn1  out A   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn2  out B   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn3  out C   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn4  out C   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mp1  out A   C1  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp2  C1  B   C2  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp3  C2  C   C3  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
mp4  C3  C   dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
.ends nor4

.subckt nand4 dd ss sub A B C D out
mn1  out A   C1  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn2  C1  B   C2  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn3  C2  C   C3  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mn4  C3  C   ss  sub  n1  w=4u  l=0.15u AS=6p AD=6p PS=7u PD=7u
mp1  out A   dd  sub  p1  w=10u l=0.15u AS=15p AD=15p PS=13u PD=13u
mp2  out B   dd  sub  p1  w=10u l=0.15u AS=15p AD=15p PS=13u PD=13u
mp3  out C   dd  sub  p1  w=10u l=0.15u AS=15p AD=15p PS=13u PD=13u
mp4  out C   dd  sub  p1  w=10u l=0.15u AS=15p AD=15p PS=13u PD=13u
.ends nand4

.subckt inv1 dd ss sub in out
mn1  out in  ss  sub  n1  w=4u  l=0.15u  AS=6p AD=6p PS=7u PD=7u
mp1  out in  dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u
.ends inv1

.subckt inv5 dd ss sub in out
xinv1 dd ss sub in 1 inv1
xinv2 dd ss sub 1  2 inv1
xinv3 dd ss sub 2  3 inv1
xinv4 dd ss sub 3  4 inv1
xinv5 dd ss sub 4 out inv1
.ends inv5

.subckt inv25 dd ss sub in out
xinv1 dd ss sub in 1 inv5
xinv2 dd ss sub 1  2 inv5
xinv3 dd ss sub 2  3 inv5
xinv4 dd ss sub 3  4 inv5
xinv5 dd ss sub 4 out inv5
.ends inv25

.subckt inv125 dd ss sub in out
xinv1 dd ss sub in 1 inv25
xinv2 dd ss sub 1  2 inv25
xinv3 dd ss sub 2  3 inv25
xinv4 dd ss sub 3  4 inv25
xinv5 dd ss sub 4 out inv25
.ends inv125

.subckt inv625 dd ss sub in out
xinv1 dd ss sub in 1 inv125
xinv2 dd ss sub 1  2 inv125
xinv3 dd ss sub 2  3 inv125
xinv4 dd ss sub 3  4 inv125
xinv5 dd ss sub 4 out inv125
.ends inv625
