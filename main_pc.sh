
# Set environmental variables
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_CONTEXT_LENGTH=20000
#export OLLAMA_KV_CACHE_TYPE=q8_0

# Set conda environment


###################################################
start_time=$SECONDS
echo " Start Time:     $(date +%T)"



###################################################
ollama serve > ollama.log 2>&1 & disown

python -u src/agent/graph.py

#pkill ollama
###################################################
end_time=$SECONDS
echo " Finish Time:      $(date +%T)"
echo " Time Taken:       $((end_time - start_time)) seconds"