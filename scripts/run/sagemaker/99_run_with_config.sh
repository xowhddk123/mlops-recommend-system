#!/bin/bash

############################################################
# Help                                                     
############################################################
Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo
   echo "Syntax: scriptTemplate [-h|n|s|t|d|j]"
   echo "options:"
   echo "-n     Namespace name."
   echo "-s     Service name."
   echo "-t     Task name."
   echo "-d     Base date."
   echo "-j     Job name."
   echo "-x     Dependency job name."
   echo
}

############################################################
# Process the input options. Add options as needed.       
############################################################
n_flag=0
s_flag=0
t_flag=0
d_flag=0
j_flag=0
x_flag=0
while getopts :n:s:t:d:j:x:h flag
do
    case "${flag}" in
        h)
          Help
          exit;;
        n)
          n_flag=1
          namespace=${OPTARG};;
        s)
          s_flag=1
          service=${OPTARG};;
        t)
          t_flag=1
          task=${OPTARG};;
        d)
          d_flag=1
          base_date=${OPTARG};;
        j)
          j_flag=1
          job=${OPTARG};;
        x)
          x_flag=1
          dependency_job=${OPTARG};;
        :)                                    # If expected argument omitted:
          echo "Error: -${OPTARG} requires an argument."
          exit_abnormal                       # Exit abnormally.
          ;;
    esac
done

if [ $n_flag -eq 0 ]
  then
    echo "Error: -n requires an argument."
    Help
  exit 2
elif [ $s_flag -eq 0 ]
  then
    echo "Error: -s requires an argument."
    Help
  exit 2
elif [ $t_flag -eq 0 ]
  then
    echo "Error: -t requires an argument."
    Help
  exit 2
elif [ $d_flag -eq 0 ]
  then
    echo "Error: -d requires an argument."
    Help
  exit 2
elif [ $j_flag -eq 0 ]
  then
    echo "Error: -j requires an argument."
    Help
  exit 2
fi

############################################################
# Process task with sagemaker                              
############################################################
SCRIPT=`readlink -f $0`
SCRIPT_PATH=`dirname ${SCRIPT}`
PROJECT_PATH=${SCRIPT_PATH}/../../..
SRC_PATH=${PROJECT_PATH}/src
CONFIG_PY=${PROJECT_PATH}/config_controller.py
ENTRY_POINT=main_with_sagemaker.py
args=`python ${CONFIG_PY} get-run-script \
  ${namespace} \
  ${service} \
  ${task} \
  ${base_date} \
  ${job} \
  ${dependency_job}`
run_script="python ${ENTRY_POINT} ${args}"

echo "====================================================="
echo "RUN SCRIPT INFO"
echo " NAMESPACE       : ${namespace}"
echo " SRC_PATH        : ${SRC_PATH}"
echo " CONFIG_PY       : ${CONFIG_PY}"
echo " ENTRY_POINT     : ${ENTRY_POINT}"
echo " SERVICE_NAME    : ${service}"
echo " TASK_NAME       : ${task}"
echo " BASE_DATE       : ${base_date}"
echo " RUN_SCRIPT      : ${run_script}"
echo " JOB             : ${job}"
echo " DEPENDENCY_JOB  : ${dependency_job}"
echo "====================================================="

eval "cd ${SRC_PATH} && ${run_script}"