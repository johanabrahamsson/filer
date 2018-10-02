let $class := doc("battleships.xml")/Ships/Class
for $x in $class
where $x/@displacement > 35000
return $x
