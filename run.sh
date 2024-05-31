#!/bin/bash
{

#conda activate genzi

export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYOPENGL_PLATFORM=egl


#============ Sketchfab Dataset ============#

# Generation
python -m genzi.generation run_cfg="config/sketchfab_gen.yml"

# Evaluation
#python -m genzi.evaluation run_cfg="config/sketchfab_eval.yml"


#============ PROX-S Dataset ============#

# Generation
#python -m genzi.generation run_cfg="config/proxs_gen.yml"

# Evaluation
#python -m genzi.evaluation run_cfg="config/proxs_eval.yml"

exit
}
