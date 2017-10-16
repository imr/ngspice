#!/bin/sh
# -*- mode: tcl -*- \
        exec wish -f "$0" ${1+"$@"}

package require BLT
load ../../../src/.libs/libspice.so

spice::source "diffpair.cir"
spice::op
spice::let Vd = V(5) - V(4)
blt::vector create imag
blt::vector create real

set ok 0
###################
puts " Vd is a real vector of length 1"
###################
#too many arguments
if {[catch {spice::vectoblt raul ibrahim ector karim} erreur] != 0} {puts "ERROR EXPECTED: TEST 1 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 1 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#no acceptable argument
if {[catch {spice::vectoblt raul ibrahim ector} erreur] != 0} {puts "ERROR EXPECTED: TEST 2 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 2 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#no acceptable blt vector
if {[catch {spice::vectoblt Vd ibrahim} erreur] != 0} {puts "ERROR EXPECTED: TEST 3 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 3 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#real part affectation
if {[catch {spice::vectoblt Vd real} erreur] == 0} {puts "NO ERROR IN AFFECTATION. TEST 4 OK:\n\t (Blank line)"; set ok [expr {$ok + 1}]} else {puts "TEST 4 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#no acceptable blt vector (2 vectors)
if {[catch {spice::vectoblt Vd ibrahim ector} erreur] != 0} {puts "ERROR EXPECTED: TEST 5 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 5 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#no acceptable imaginary vector
if {[catch {spice::vectoblt Vd real ector} erreur] != 0} {puts "ERROR EXPECTED: TEST 6 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 6 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#real and imaginary part affectation
if {[catch {spice::vectoblt Vd real imag} erreur] == 0} {puts "NO ERROR IN AFFECTATION. TEST 7 OK:\n\t (Blank line)"; set ok [expr {$ok + 1}]} else {puts "TEST 7 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#all good vectors, but another argument invited himself
if {[catch {spice::vectoblt Vd real imag karim} erreur] != 0} {puts "ERROR EXPECTED: TEST 8 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 8 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}

###################
puts " Vd is a complex vector of length 10"
###################
spice::op
spice::ac dec 10 100 1000
spice::let Vd = V(5) - V(4)
#too many arguments
if {[catch {spice::vectoblt raul ibrahim ector karim} erreur] != 0} {puts "ERROR EXPECTED: TEST 1 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 1 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#no acceptable argument
if {[catch {spice::vectoblt raul ibrahim ector} erreur] != 0} {puts "ERROR EXPECTED: TEST 2 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 2 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#no acceptable blt vector
if {[catch {spice::vectoblt Vd ibrahim} erreur] != 0} {puts "ERROR EXPECTED: TEST 3 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 3 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#real part affectation
if {[catch {spice::vectoblt Vd real} erreur] == 0} {puts "NO ERROR IN AFFECTATION. TEST 4 OK:\n\t (Blank line)"; set ok [expr {$ok + 1}]} else {puts "TEST 4 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#no acceptable blt vector (2 vectors)
if {[catch {spice::vectoblt Vd ibrahim ector} erreur] != 0} {puts "ERROR EXPECTED: TEST 5 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 5 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#no acceptable imaginary vector
if {[catch {spice::vectoblt Vd real ector} erreur] != 0} {puts "ERROR EXPECTED: TEST 6 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 6 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#real and imaginary part affectation
if {[catch {spice::vectoblt Vd real imag} erreur] == 0} {puts "NO ERROR IN AFFECTATION. TEST 7 OK:\n\t (Blank line)"; set ok [expr {$ok + 1}]} else {puts "TEST 7 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}
#all good vectors, but another argument invited himself
if {[catch {spice::vectoblt Vd real imag karim} erreur] != 0} {puts "ERROR EXPECTED: TEST 8 OK:\n\t$erreur"; set ok [expr {$ok + 1}]} else {puts "TEST 8 ERROR\n\t  Test 1 should return TCL_ERROR, but it does not"}

puts "\n\n"
if {$ok == 16} {puts "spice::vectoblt OK ($ok/16 tests ok)"} else {puts "spice::vectoblt KO:\n\t$ok/16 tests ok"}
