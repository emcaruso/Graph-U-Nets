SCRIPTDIR=$(dirname "$0")
MAINDIR=${SCRIPTDIR}"/Method/main.py"

for batch in 1 3
do
for n_gcn in 1 3 5
do
for l_dim in 32 64
do

python3 $MAINDIR -l_dim $l_dim -n_gcn $n_gcn -batch $batch

done
done
done


