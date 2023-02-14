SCRIPTDIR=$(dirname "$0")
MAINDIR=${SCRIPTDIR}"/Method/main.py"

for lr in 0.001 0.01 0.0001
do
for batch in 3 5 8
do
for l_dim in 16 32 48 64
do
for drop_n in 0.3 0.2 0.4
do

python3 $MAINDIR -lr $lr -batch $batch -l_dim $l_dim -drop_n $drop_n

done
done
done
done


