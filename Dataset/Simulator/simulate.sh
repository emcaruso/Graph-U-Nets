SCRIPTDIR=$(dirname "$0")
SIMULATIONDIR=${SCRIPTDIR}"/thermomechanical_simulation/simulation/"
EXPORTFILE=${SIMULATIONDIR}"export"
MODELSDIR=${SCRIPTDIR}"/../Data/3d_models/"
MESHESDIR=${MODELSDIR}"med_meshes/"
MODELSDIR="$(echo $MODELSDIR | realpath -s ${MODELSDIR})"
MODELSDIR="$(echo $MODELSDIR | sed 's/\//\\\//g')"
SIMULATIONDIR="$(echo $SIMULATIONDIR | realpath -s ${SIMULATIONDIR})"
SIMULATIONDIR="$(echo $SIMULATIONDIR | sed 's/\//\\\//g')"

MEDRESDIR=${SCRIPTDIR}"/../Data/3d_models/med_results/"
TABLESDIR=${SCRIPTDIR}"/../Data/3d_models/table_results/"

MEDRESDIR="$(echo ${MEDRESDIR} | realpath -s ${MEDRESDIR})"
TABLESDIR="$(echo ${TABLESDIR} | realpath -s ${TABLESDIR})"

# eliminate old results
rm -rv $MEDRESDIR
mkdir $MEDRESDIR
rm -rv $TABLESDIR
mkdir $TABLESDIR

# iterate over med files
for filepath in ${MESHESDIR}*
do
	medname=${filepath##*/}
	name=${medname::-4}
	# TODO control on format
	# replace line in expert file
	sed -i "19s/.*/F comm $SIMULATIONDIR\/RunCase_1_Stage_3.comm D  1/" ${EXPORTFILE}
	sed -i "20s/.*/F libr $MODELSDIR\/med_meshes\/$medname D  3/" ${EXPORTFILE}
	sed -i "21s/.*/F libr $MODELSDIR\/med_results\/thrm_$medname  R  2/" ${EXPORTFILE}
	sed -i "22s/.*/F libr $MODELSDIR\/med_results\/meca_$medname  R  4/" ${EXPORTFILE}
	sed -i "23s/.*/F libr $MODELSDIR\/table_results\/$name.txt R  8/" ${EXPORTFILE}
	sed -i "24s/.*/F mess $SIMULATIONDIR\/message R  6/" ${EXPORTFILE}
	# run simulation
	# echo "as_run $EXPORTFILE" | /home/emanuelecaruso/Desktop/salome-meca_program/salome_meca-lgpl-2021.1.0-2-20220817-scibian-9 shell
	echo "as_run $EXPORTFILE" | /home/manu/Desktop/salome-meca_program/salome_meca-lgpl-2021.1.0-2-20220817-scibian-9 shell
done

$SCRIPTDIR"/../Dataset_blender/generate_surfaces_out.sh"
