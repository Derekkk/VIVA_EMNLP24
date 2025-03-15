import json
import os
import re
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer, scoring
from nltk.tokenize import word_tokenize
import json
from google_bleu import compute_bleu
import numpy as np
import string
import collections
from nltk.util import ngrams
from meteor import meteor_score
from bert_score import score
from evaluate import load
# from pycocoevalcap.cider.cider import Cider
from evaluate import load
from bert_score import score as score_bert
import openai
import time
import ast
import random
random.seed(1234)


# bertscore = load("bertscore")

# cider_score = Cider()
# mauve = load('mauve')


def openai_generate(input_prompt, model="gpt-3.5-turbo-0125", temperature=1):
    API_KEY = "" # Your Openai
    for _ in range(5):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_prompt}
                ],
                temperature=temperature,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                api_key=API_KEY,
                #api_base = API_BASE
            )
            break
        except Exception as e:
            print(["[OPENAI ERROR]: ", e])
            response = None
            time.sleep(2)
    if response != None:
    # print(response)
        response = response.choices[0].message.content
    return response


def eval_gptscore_one(mcq_answer, gen, ref):
    prompt_template = f'''
Background: You are an impartial judge. There is a multiple-choice question about selecting the most appropriate action to handle a situation, the correct answer, and a gold-standard explanation of why selecting this answer. You will also be provided with a model generated explanation.

Your task is to evaluate the quality of a generated explanation compared to the gold-standard explanation. Then, assign a score on a scale of 1 to 5 based on its quality, where 1 is the lowest and 5 is the highest. Specific Evaluation Criteria:
- 1: The model-generated explanation significantly deviates from the gold-standard explanation and fails to address the correct rationale;
- 3: The model-generated explanation captures most of the key points found in the gold-standard explanation, but some important aspects are missing or inaccurately represented;
- 5: The model-generated explanation accurately covers all key points present in the gold-standard explanation.

Now please give a score based on the content:

- [multiple-choice question]:
{mcq_answer}

- [gold-standard explanation]:
{ref}

- [model-generated explanation]:
{gen}

Please directly output a score by strictly following this format: [[score]], for example: Rating: [[3]].
'''
    judgment = openai_generate(prompt_template.strip())
    # print(f"[log-output]: {[judgment]}")
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    # print("[log-llm_judge_eval-judgement]: ", [judgment])
    if judgment == None:
        return 0
    match = re.search(one_score_pattern, judgment)
    if not match:
        match = re.search(one_score_pattern_backup, judgment)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = 0
    return rating



def eval_meteor(gen_list, ref_list):
    score_list = []
    for gen, ref in zip(gen_list, ref_list):
        score = round(meteor_score([" ".join(ref)], " ".join(gen)),4)
        score_list.append(score)
    # return np.mean(score_list)
    # print("meteor score: ", np.mean(score_list))
    return  np.mean(score_list)


def eval_rouge(gen_list, ref_list):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    for gen, ref in zip(gen_list, ref_list):
        gen_str = " ".join(gen)
        ref_str = " ".join(ref)
        cur_score = scorer.score(ref_str, gen_str)
        aggregator.add_scores(cur_score)

    aggregates = aggregator.aggregate()
    rouge_socre_dict = {}
    for score_type, aggregate in sorted(aggregates.items()):
        # print("%s-R,%f,%f,%f\n" %
        #      (score_type, aggregate.low.recall, aggregate.mid.recall,
        #      aggregate.high.recall))
        # print("%s-P,%f,%f,%f\n" %
        #      (score_type, aggregate.low.precision,
        #      aggregate.mid.precision, aggregate.high.precision))
        # print("%s-F,%f,%f,%f\n" %
        #      (score_type, aggregate.low.fmeasure,
        #      aggregate.mid.fmeasure, aggregate.high.fmeasure))
        rouge_socre_dict[score_type] = {
            "p": aggregate.mid.precision, 
            "r": aggregate.mid.recall, 
            "f": aggregate.mid.fmeasure
        }
    return rouge_socre_dict


def eval_bleu(gen_list, ref_list):
    reference_corpus = [[elem] for elem in ref_list]
    translation_corpus = gen_list

    bleu_score_dict = {}
    for max_order in [1, 2, 3, 4]:
        bleu_score = compute_bleu(
            reference_corpus, 
            translation_corpus, 
            max_order=max_order,
            smooth=True
            )
        bleu_score_dict[max_order] = bleu_score[0]
    
    chencherry = SmoothingFunction()
    nltk_bleu = corpus_bleu(
        reference_corpus, 
        translation_corpus, 
        smoothing_function=chencherry.method1
        )
    bleu_score_dict["nltk"] = nltk_bleu
    
    # return bleu_score_dict
    # for score_type in bleu_score_dict:
    #     print("bleu-", score_type, ": ", bleu_score_dict[score_type])
    return bleu_score_dict


def get_bertscore(cands, refs):
    P1, R1, F1 = score_bert(cands, refs, lang="en", 
    model_type='microsoft/deberta-large-mnli',
    device="cuda")
    # P1, R1, F1 = score_bert(cands, refs, lang="en", device="cuda")
    F1 = F1.cpu().detach().numpy().tolist()
    P1 = P1.cpu().detach().numpy().tolist()
    R1 = R1.cpu().detach().numpy().tolist() 
    return F1, P1, R1
            

def eval_one_gen_batch(refs, preds, mcq_answers):
    # print("\n=======evaluate bert score=======")
    bert_f1s, bert_p1s, bert_r1s = get_bertscore(preds, refs)

    bleurt = load("bleurt", "BLEURT-20", module_type="metric")
    bleurt_score_list = bleurt.compute(predictions=preds, references=refs)["scores"]

    gpt_scores = []
    for mcq_answer, ref, gen in zip(mcq_answers, refs, preds):
        cur_score = eval_gptscore_one(mcq_answer, gen, ref) 
        # cur_score = 0
        gpt_scores.append(cur_score)

    results = []
    for bert_f1, bert_p1, bert_r1, bleurt_score, gpt_score in zip(bert_f1s, bert_p1s, bert_r1s, bleurt_score_list, gpt_scores):
        bert_f1 = bert_f1 * 100
        bert_p1 = bert_p1 * 100
        bert_r1 = bert_r1 * 100
        bleurt_score = bleurt_score * 100
        average_score = np.mean([bert_f1, bleurt_score])
        result_dict = {"bert_p1": bert_p1, "bert_r1": bert_r1, "bert_score": bert_f1, "bleurt_score": bleurt_score, "explantaion_score": average_score, "gpt_score": gpt_score}
        results.append(result_dict)
    
    return results

def eval_one_gen(ref, pred):
    # print("\n=======evaluate bert score=======")
    bert_score = get_bertscore([pred], [ref])[0] * 100

    bleurt = load("bleurt", "BLEURT-20", module_type="metric")
    bleurt_score = bleurt.compute(predictions=[pred], references=[ref])["scores"][0]  * 100

    average_score = np.mean([bert_score, bleurt_score])

    result_dict = {"bert_score": bert_score, "bleurt_score": bleurt_score, "explantaion_score": average_score}
    return result_dict


def generate_eval_file(file_path):
    data = json.load(open(file_path))
    data = data[:100]

    refs = []
    preds = []
    mcq_answers = []

    for sample in data:
        # norm
        ref = sample["rationale"]
        # ref = reason_sample["rationale"]
        pred = sample["result"][0]["prediction"]
        if pred is None or ref is None:
            continue
        if "USER" in pred and "ASSISTANT:" in pred:
            pred = pred.split("ASSISTANT:")[1].strip()
        preds.append(pred)
        refs.append(ref)
        mcq_answers.append(sample["mcq_answer"])
    
    print(f"num preds: {len(preds)}, refs: {len(refs)}")
    refs_tokens = [len(word_tokenize(elem)) for elem in refs]
    pred_tokens = [len(word_tokenize(elem)) for elem in preds]
    print("avg. length: ", np.mean(pred_tokens), np.mean(refs_tokens))

    # bertscore
    P1, R1, F1 = score_bert(preds, refs, lang="en", 
        model_type='microsoft/deberta-large-mnli',
        device="cuda")
    P1 = P1.cpu().detach().numpy().tolist()
    R1 = R1.cpu().detach().numpy().tolist() 
    F1 = F1.cpu().detach().numpy().tolist()

    # BLUERT
    bleurt = load("bleurt", "BLEURT-20", module_type="metric")
    bleurt_score = bleurt.compute(predictions=preds, references=refs)["scores"]

    # GPT
    gpt_scores = []
    for mcq_answer, ref, gen in zip(mcq_answers, refs, preds):
        cur_score = eval_gptscore_one(mcq_answer, gen, ref) 
        
        gpt_scores.append(cur_score)

    score_dict = {
        "BERTScore": {"p": np.mean(P1), "r": np.mean(R1), "f": np.mean(F1)},
        "BLEURT": np.mean(bleurt_score),
        "gpt_scores": np.mean(gpt_scores)
    }
    return score_dict


def gptscore_eval(file_path):
    data = json.load(open(file_path))
    refs = []
    preds = []
    mcq_answers = []

    for sample in data:
        # norm
        ref = sample["rationale"]
        # ref = reason_sample["rationale"]
        pred = sample["result"][0]["prediction"]
        if pred is None or ref is None:
            continue
        if "USER" in pred and "ASSISTANT:" in pred:
            pred = pred.split("ASSISTANT:")[1].strip()
        preds.append(pred)
        refs.append(ref)
        mcq_answers.append(sample["action_answer"])
    
    gpt_scores = []
    for mcq_answer, ref, gen in zip(mcq_answers, refs, preds):
        cur_score = eval_gptscore_one(mcq_answer, gen, ref) 
        # cur_score = 0
        gpt_scores.append(cur_score)
    return gpt_scores


if __name__ == "__main__":
    reason_folder = "../reason_generation"
    file_list = [elem for elem in os.listdir(reason_folder)]

    gpt_score_dict = {}
    for file_name in file_list:
        print("===" * 10)
        print(file_name)
        cur_path = reason_folder + "/" + file_name
        result = generate_eval_file(cur_path)
        print(result)
        print()
        
    
   