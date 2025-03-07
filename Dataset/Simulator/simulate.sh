SCRIPTDIR=$(dirname "$0")
SIMULATIONDIR=${SCRIPTDIR}"/thermomechanical_simulation/simulation/"
EXPORTFILE=${SIMULATIONDIR}"export"
COMMFILE=${SIMULATIONDIR}"RunCase_1_Stage_3.comm"


############ CHOOSE DATASET #############

DATA="synth_vol"
#DATA="synth"
#DATA="real_vol"
#DATA="real"

#########################################


MODELSDIR=${SCRIPTDIR}"/../Data/"$DATA
MODELSDIR="$(echo $MODELSDIR | realpath -s ${MODELSDIR})"
MODELSDIR="$(echo $MODELSDIR | sed 's/\//\\\//g')"

list_patches="patch_0 patch_1 patch_2 patch_3 patch_4 patch_5 patch_6 patch_7 patch_8 patch_9"
#list_fluxes='100000 300000 500000'
list_fluxes='500000'



for i in $list_patches; do
	sed -i "28s/.*/                                         GROUP_MA=('$i', )),/" ${COMMFILE}
	for j in $list_fluxes; do
		medname=$j,$i
		
		sed -i "27s/.*/                             FLUX_REP=_F(FLUN=$j.0,/" ${COMMFILE}
		sed -i "20s/.*/F libr $MODELSDIR\/$DATA.med  D  20/" ${EXPORTFILE}
		sed -i "21s/.*/F libr $MODELSDIR\/med_results\/thrm_$medname.med  R  3/" ${EXPORTFILE}
		sed -i "22s/.*/F libr $MODELSDIR\/med_results\/meca_$medname.med  R  2/" ${EXPORTFILE}
		sed -i "23s/.*/F libr $MODELSDIR\/table_results\/$medname.txt R  8/" ${EXPORTFILE}
		
		echo "as_run $EXPORTFILE" | /home/manu/Desktop/salome-meca_program/salome_meca-lgpl-2021.1.0-2-20220817-scibian-9 shell
	done
done

#echo "as_run $EXPORTFILE" | /home/manu/Desktop/salome-meca_program/salome_meca-lgpl-2021.1.0-2-20220817-scibian-9 shell
