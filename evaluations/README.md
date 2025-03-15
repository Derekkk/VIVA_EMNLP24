## This folder contains scripts for Evaluations

- eval_script/: contains scripts for evaluating both action selection, value inference and reason generation
- action_selection/: contains sample outputs for the action selection task (evaluated by accuracy)
- value_inference/: contains sample outputs for the value inference task (evaluated by accuracy)
- reason_generation/: contains sample outputs for the reason generation task (evaluated with text generation metrics)

Specifically:
1. Change your OpenAI key in evaluation.py | evaluation_gen.py
2. Specify the task and result folder

- For action selection and value infernece:
  - change the task settings (action|value)

```
cd eval_script
python3 evaluation.py
```

- For reason generation:
```
cd eval_script
python3 evaluation_gen.py
```
