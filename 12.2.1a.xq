let $printers := doc("products.xml")/Products/Maker/Printer
for $x in $printers
where $x/@price le '100'
return $x

