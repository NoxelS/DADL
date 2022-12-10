#!/bin/tcsh -f



foreach i (`seq 1 50`)
  awk -v i=$i 'i==NR {print}' test 
#  echo $i  
end

