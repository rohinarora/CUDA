#set -o xtrace
./vAdd 10000000 32
./vAdd 10000000 64
./vAdd 10000000 128
./vAdd 10000000 256
./vAdd 10000000 512
./vAdd 10000000 1024
./vAdd 10000000 2048 #will give error. maybe catch via memcheck, or checking cuda return flags
./vAdd 1000000 1024
./vAdd 100000 1024
./vAdd 10000 1024
./vAdd 1000 1024
./vAdd 100 1024
./vAdd 10 1024
