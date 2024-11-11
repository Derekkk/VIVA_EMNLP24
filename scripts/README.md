## This folder contains scripts for model prediction.

For each model, modify the command:

```
#Task claude3 as an example
#task options: mcq | norm | reason_generation

data="data/VIVA_annotation.json"
image_folder="data/VIVA_images/"
write_path=""
task=""

python3 -u predict_claude_opus.py \
    --read_path ${data} \
    --write_path "results/results_claude3_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}

```


### Experimental Setting
- Sample components: (image, actions, norms, reason)

#### Level-1 Task on Action Selection
- p(answer|image, actions)
  - Input: image + actions
  - Output: the selected action
- task parameter: ```mcq```
  
#### Level-2 Task on Value Inference
- p(label|image, action_answer, value)
	- Input: image + action_answer + value
 	- Output: relevant / not relevant
- task parameter: ```norm```

#### Level-2 Task on Reason Generation
- p(reason|image, action_answer)
	- Input: image + action_answer
 	- Output: reason 
- task parameter: ```reason_generation```
