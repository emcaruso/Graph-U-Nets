SCRIPTDIR=$(dirname "$0")
BLENFILE=${SCRIPTDIR}"/scene_dataset.blend"
PYFILE_CREATE_DEFORMATIONS=${SCRIPTDIR}"/generate_deformed_shapes.py"
PYFILE_GENERATE_INPUTS=${SCRIPTDIR}"/generate_inputs_for_NN.py"

echo $PYDIR

blender --background ./${BLENFILE} --python ${PYFILE_CREATE_DEFORMATIONS}
#blender --background ./${BLENFILE} --python ${PYFILE_GENERATE_INPUTS}
