#!/bin/bash

wget -O example-1.tar.xz https://www.dropbox.com/s/2zva3f0yoi0i0j5/example-1.tar.xz?dl=0
wget -O example-2.tar.xz https://www.dropbox.com/s/aups7uc2xxbbshb/example-2.tar.xz?dl=0
wget -O example-3.tar.xz https://www.dropbox.com/s/45tokme57rbiz6e/example-3.tar.xz?dl=0

tar -xJvf example-1.tar.xz
tar -xJvf example-2.tar.xz
tar -xJvf example-3.tar.xz
