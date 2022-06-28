FILE=$1
URL=http://xxx.edu/models/$FILE.h5
MODEL_FILE=./models/$FILE.h5
wget -N $URL -O $MODEL_FILE