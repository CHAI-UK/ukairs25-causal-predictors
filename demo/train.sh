CASES=("TC-conf")
CONFS=("observed")

for CASE in ${CASES[@]}
do
    for CONF in ${CONFS[@]}
    do
        python ./train_models.py -o ./models --case $CASE --conf $CONF
    done
done