Voice Selector

Este programa usa una red neural entrenada para reconocer lo que son voces de los que no.

El proposito de este programa es la de seleccionar una seccion de un audio que sea la voz de un humano
y no ruido o vacio.

Para compilar puede usar el comando "make" que se encargara de ello.
De no poseer las librerias necesarias, corra el comando "bash install.sh" y se instalaran las librerias
El makefile supone que haskell esta instalado en la computadora, de no estarlo, correr "sudo apt-get install haskell-platform"

Los dos programas selectVoice y selectVoiceFast se corren de la misma manera, pero selectVoiceFast esta optimizado para correr mas rapido
puede comparar el tiempo con el comando "bash compara.sh <archivo .wav>"
Estos dos programas procesan los audios provistos y retornan un archivo con las n decimas de segundos mas humanos, n es el segundo argumento

para correr cualquiera de los dos programas correr el comando:
./selectVoice <archivo de entrada (.wav)> <un numero que sera la cantidad de decimas de segundo a seleccionar> <nombre archivo de salida>
