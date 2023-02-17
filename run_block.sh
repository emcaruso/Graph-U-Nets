SCRIPTDIR=$(dirname "$0")
MAINDIR=${SCRIPTDIR}"/Method/main.py"

for lr in 0.001 0.0001
do
for l_dim in 16 32 48 64
do
for drop_n in 0.2 0.4
do
for act_o in 'ReLU' 'Sigmoid'

python3 $MAINDIR -lr $lr -l_dim $l_dim -drop_n $drop_n -act_o act_o

done
done
done


