let $shipname := doc("battleships.xml")/Ships/Class/Ship/@name
return data($shipname)
