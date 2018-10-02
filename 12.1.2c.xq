let $class := doc("battleships.xml")/Ships/Class
for $x in $class
where $x/Ship/@launched < 1917
return $x/Ship
