#!/bin/bash

wget -O example-1.tar.xz https://www.dropbox.com/s/rq4l7925pw8ac0x/example-1.tar.xz?dl=0
wget -O example-2.tar.xz https://www.dropbox.com/s/i43cup4d6ftjieb/example-2.tar.xz?dl=0
wget -O example-3.tar.xz https://www.dropbox.com/s/lo72s1m35rmlvw0/example-3.tar.xz?dl=0

tar -xJvf example-1.tar.xz
tar -xJvf example-2.tar.xz
tar -xJvf example-3.tar.xz
