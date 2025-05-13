CASES=("3-conf")
CONFS=("observed" "hidden")

for CASE in ${CASES[@]}
do
    for CONF in ${CONFS[@]}
    do
        python ./train_models.py -o ./models --case $CASE --conf $CONF
    done
done