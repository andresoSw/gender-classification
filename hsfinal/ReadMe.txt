Calcula

Este programa usa una red neural entrenada para reconocer el genero en un archivo de audio.

Para compilar puede usar el comando "make" que se encargara de ello.
De no poseer las librerias necesarias, corra el comando "bash install.sh" y se instalaran las librerias
El makefile supone que haskell esta instalado en la computadora, de no estarlo, correr "sudo apt-get install haskell-platform"

Para correr este programa corra:
./calcula <archivo .wav>
El programa imprimira en pantalla la probabilidad de que este sea hobre o mujer,
donde mientras mas cerca a 1 se considera que es hombre y mientras mas cerca a 0 se considera que es mujer 
