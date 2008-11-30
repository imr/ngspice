# differentiate.tcl --
#    Numerical differentiation
#

namespace eval ::math::calculus {
}
namespace eval ::math::optimize {
}

# deriv --
#    Return the derivative of a function at a given point
# Arguments:
#    func         Name of a procedure implementing the function
#    point        Coordinates of the point
#    scale        (Optional) the scale of the coordinates
# Result:
#    List representing the gradient vector at the given point
# Note:
#    The scale is necessary to create a proper step in the
#    coordinates. The derivative is estimated using central
#    differences.
#    The function may have an arbitrary number of arguments,
#    for each the derivative is determined - this results
#    in a list of derivatives rather than a single value.
#    (No provision is made for the function to be a
#    vector function! So, second derivatives are not
#    possible)
#
proc ::math::calculus::deriv {func point {scale {}} } {

   set epsilon 1.0e-12
   set eps2    [expr {sqrt($epsilon)}]

   #
   # Determine a scale
   #
   foreach c $point {
      if { $scale == {} } {
         set scale [expr {abs($c)}]
      } else {
         if { $scale < abs($c) } {
            set scale [expr {abs($c)}]
         }
      }
   }
   if { $scale == 0.0 } {
      set scale 1.0
   }

   #
   # Check the number of coordinates
   #
   if { [llength $point] == 1 } {
      set v1 [$func [expr {$point+$eps2*$scale}]]
      set v2 [$func [expr {$point-$eps2*$scale}]]
      return [expr {($v1-$v2)/(2.0*$eps2*$scale)}]
   } else {
      set result {}
      set idx    0
      foreach c $point {
         set c1 [expr {$c+$eps2*$scale}]
         set c2 [expr {$c-$eps2*$scale}]

         set v1 [eval $func [lreplace $point $idx $idx $c1]]
         set v2 [eval $func [lreplace $point $idx $idx $c2]]

         lappend result [expr {($v1-$v2)/(2.0*$eps2*$scale)}]
         incr idx
      }
      return $result
   }
}

# auxiliary functions --
#
proc ::math::optimize::unitVector {vector} {
   set length 0.0
   foreach c $vector {
      set length [expr {$length+$c*$c}]
   }
   scaleVector $vector [expr {1.0/sqrt($length)}]
}
proc ::math::optimize::scaleVector {vector scale} {
   set result {}
   foreach c $vector {
      lappend result [expr {$c*$scale}]
   }
   return $result
}
proc ::math::optimize::addVector {vector1 vector2} {
   set result {}
   foreach c1 $vector1 c2 $vector2 {
      lappend result [expr {$c1+$c2}]
   }
   return $result
}

# minimumSteepestDescent --
#    Find the minimum of a function via steepest descent
#    (unconstrained!)
# Arguments:
#    func         Name of a procedure implementing the function
#    point        Coordinates of the starting point
#    eps          (Optional) measure for the accuracy
#    maxsteps     (Optional) maximum number of steps
# Result:
#    Coordinates of a point near the minimum
#
proc ::math::optimize::minimumSteepestDescent {func point {eps 1.0e-5} {maxsteps 100} } {

   set factor  100
   set nosteps 0
   if { [llength $point] == 1 } {
      while { $nosteps < $maxsteps } {
         set fvalue   [$func $point]
         set gradient [::math::calculus::deriv $func $point]
         if { $gradient < 0.0 } {
            set gradient -1.0
         } else {
            set gradient  1.0
         }
         set found    0
         set substeps 0
         while { $found == 0 && $substeps < 3 } {
            set newpoint [expr {$point-$factor*$gradient}]
            set newfval  [$func $newpoint]

            #puts "factor: $factor - point: $point"
            #
            # Check that the new point has a lower value for the
            # function. Can we increase the factor?
            #
            #
            if { $newfval < $fvalue } {
               set point     $newpoint

#
#               This failed with sin(x), x0 = 1.0
#               set newpoint2 [expr {$newpoint-$factor*$gradient}]
#               set newfval2  [$func $newpoint2]
#               if { $newfval2 < $newfval } {
#                  set factor [expr {2.0*$factor}]
#                  set point  $newpoint2
#               }
               set found 1
            } else {
               set factor [expr {$factor/2.0}]
            }

            incr substeps
         }

         #
         # Have we reached convergence?
         #
         if { abs($factor*$gradient) < $eps } {
            break
         }
         incr nosteps
      }
   } else {
      while { $nosteps < $maxsteps } {
         set fvalue   [eval $func $point]
         set gradient [::math::calculus::deriv $func $point]
         set gradient [unitVector $gradient]

         set found    0
         set substeps 0
         while { $found == 0 && $nosteps < $maxsteps } {
            set newpoint [addVector $point [scaleVector $gradient -$factor]]
            set newfval  [eval $func $newpoint]

            #puts "factor: $factor - point: $point"
            #
            # Check that the new point has a lower value for the
            # function. Can we increase the factor?
            #
            #
            if { $newfval < $fvalue } {
               set point $newpoint
               set found 1
            } else {
               set factor [expr {$factor/2.0}]
            }

            incr nosteps
         }

         #
         # Have we reached convergence?
         #
         if { abs($factor) < $eps } {
            break
         }
         incr nosteps
      }
   }

   return $point
}

