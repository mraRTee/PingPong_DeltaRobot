# PingPong_DeltaRobot
Pingpongozó DeltaRobot projekt - vezérlés &amp; képfelismerés

A projekt egy másfél éves önálló feladat, aminek a célja egy olyan asztaliteniszező robot megépítése amely képes egy emberrel valós időben játszani.

Repository tartalmazza a delta robot 3 vezérlő motor vezérlő algoritmusát léptetőmotorral és szervóval, valamint a supervised machinelearning, képfelismerésből származó tanító algoritmus adatgyűjtés megoldását. 

A motorokat egy ATmega64 processzor vezérli, de később egy STM32 boardra szeretném átültetni.
A képfelismerés 2 kamerával történik. A pálya predikció a játékos általelütött labda indulási pontja és a korábban általam szintén 2 kamerával felvett videókból számrazó tanító adatok alapján történik.

A motor vezérlés C nyelven íródik, a képfelismerés és machinlearning rész Pythonban.

Működés:
  2 kamera által rögzített képek feldolgozása folyamatos, valós idejű. Ebből egy számítógépen futó Python program előrejelzi a várható labda érkezési koordinátákat. Ezeket a koordinátákat át kell alakítani a robot számára értelmezhető adattá. Ezt az inverz kinematika számítással lehet megtenni. Ez a számítás megadja, hogy mekkora szöggel kell elfordulnia a 3 motornak, ahhoz, hogy abba a koordinátába érkezzen a robot megfogólyában lévő pingpong ütő. A kiszámított 3 szögelfordulás értéket a számítógépről UART-on átküldi a mikrokontrollernek. A kapott értékek alapján, a 3 motor elfordul a megfelelő szögben, ezáltal a robot ütője eléri a labdát és visszaüti azt, és visszaáll eredeti állapotba. 
  
