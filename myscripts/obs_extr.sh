source src/rai_finetune/setup_finetune_shell.sh


OBS_DATA="gpt4o_observations.jsonl"
TRAINING_DATA="training_data.jsonl"

# extract the observations from the models
# python src/rai_finetune/rai_finetune/langfuse_obs_extractor.py \
#   --models "gpt-4o" "gpt-4o-mini" \
#   --output $OBS_DATA

# format the observations into the training data format
python src/rai_finetune/rai_finetune/data/training_data_formatter.py \
  --input $OBS_DATA \
  --output $TRAINING_DATA \
  --format unsloth

head -50 $TRAINING_DATA | grep -o '"tool_calls"' | wc -l