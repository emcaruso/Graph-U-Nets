SCRIPTDIR=$(dirname "$0")
MAINDIR=${SCRIPTDIR}"/Method/main.py"

for l_dim in 16 32 64 128
do
for drop_n in 0.2 0.4
do
for n_gcn in 1 3 5
do
for batch in 1 3 5
do

python3 $MAINDIR -l_dim $l_dim -drop_n $drop_n -n_gcn $n_gcn -batch $batch

done
done
done
done


