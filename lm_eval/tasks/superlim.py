from . common import HFTask, yesno
from ..utils import general_detokenize
from ..metrics import mean, acc_all, metric_max_over_ground_truths
import os
import numpy as np
from lm_eval.base import rf
from lm_eval.base import MultipleChoiceTask

class SweWsc(HFTask):
   VERSION = 0
   DATASET_PATH = "AI-Sweden/SuperLim"
   DATASET_NAME = "SweWsc"
   DATA_FILES = {"test":"SweWsc/test.csv"}
   #USE_AUTH_TOKEN = os.environ['HF_TOKEN']

   def has_training_docs(self):
        return False

   def has_validation_docs(self):
        return False

   def has_test_docs(self):
        return True

   def fewshot_description(self):
    return "Does the noun match the pronoun?"

   def doc_to_text(self, doc):
    raw_passage = doc["passage"]
    # NOTE: HuggingFace span indices are word-based not character-based.
    pre = raw_passage[:doc["challenge_begin"]]
    post = raw_passage[len(pre) + len(doc["challenge_text"]):]
    passage = pre + "*{}*".format(doc["challenge_text"]) + post
    noun = doc["response_text"]
    pronoun = doc["challenge_text"]
    text = (
            f"Passage: {passage}\n"
            + f"Fråga: I avsnittet ovan, syftar pronomenet *{pronoun}* på *{noun}*?\n"
            + "Svar:"
        )
    return text

   def doc_to_target(self, doc):
    return ' Ja' if doc['label']==1 else ' Nej'

   def construct_requests(self, doc, ctx):

        ll_yes, _ = rf.loglikelihood(ctx, ' Ja')
        ll_no, _ = rf.loglikelihood(ctx, ' Nej')

        return ll_yes, ll_no

   def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = True if doc['label'] == 1 else False
        #gold = doc['responses'][0]["correct"]

        acc = 1. if (ll_yes > ll_no) == gold else 0.

        return {
            "acc": acc
        }

   def higher_is_better(self):
        return {
            "acc": True
        }

   def aggregation(self):
        return {
            "acc": mean
        }


class SweSat(HFTask, MultipleChoiceTask):
   VERSION = 0
   DATASET_PATH = "AI-Sweden/SuperLim"
   DATASET_NAME = "SweSat"
   DATA_FILES = {"test":"SweSat/test.csv"}
   #USE_AUTH_TOKEN = os.environ['HF_TOKEN']

   def has_training_docs(self):
        return False

   def has_validation_docs(self):
        return False

   def has_test_docs(self):
        return True

   def fewshot_description(self):
    return "Which one of the alternatives is the correct definition/synonym?"

   def doc_to_text(self, doc):
    target_word = doc["target_item"]
    alternatives_list = [doc[key] for key in ["answer_1", "answer_2", "answer_3", "answer_4", "answer_5"]]
    alternatives_texts = [entry.split("/")[0] for entry in alternatives_list ]
    text = (
            f"Hitta synonym eller definition till: {target_word}\n"
            + f'Alternativer: {", ".join(str(x) for x in alternatives_texts)}\n'
            +f'Rätt svar:'
        )
    return text

   def doc_to_target(self, doc):
    alternatives = [doc[key] for key in ["answer_1", "answer_2", "answer_3", "answer_4", "answer_5"]]
    correct_answer = [alt.split("/")[0] for alt in alternatives if "1" in alt][0]
    return " " + correct_answer


   def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0]
            for alt.split("/")[0] in doc[key] for key in ["answer_1", "answer_2", "answer_3", "answer_4", "answer_5"]
        ]

        return lls
   #to fix
   def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1. if np.argmax(results) == gold else 0.
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1. if np.argmax(results / completion_len) == gold else 0.

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }
   #to fix
   def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
        }
   #to fix
   def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
        }

class SweFracas(HFTask, MultipleChoiceTask): #The json contains list and I think it cannot handle this...
   VERSION = 0
   DATASET_PATH = "AI-Sweden/SuperLim"
   DATASET_NAME = "SweFracas"
   DATA_FILES = {"test":"SweFracas/swefracas.csv"}
   #USE_AUTH_TOKEN = os.environ['HF_TOKEN']

   def has_training_docs(self):
        return False

   def has_validation_docs(self):
        return False

   def has_test_docs(self):
        return True

   def fewshot_description(self):
    return "Är svaret på frågan ja, vet ej eller nej givet de givna påståendena?"

   def _convert_standard(self, doc):
       query = self.doc_to_query(doc)
       out_doc = {
           "query": query,
           "choices": ['Ja','Vet ej','Nej'],
           "gold": ['Ja','Vet ej','Nej'].index(doc['svar'])
       }
       return out_doc

   def doc_to_query(self, doc):
    # raw_passage = doc["passage"]
    # # NOTE: HuggingFace span indices are word-based not character-based.
    # pre = " ".join(raw_passage.split()[:doc["challenge_begin"]])
    # post = raw_passage[len(pre) + len(doc["challenge_text"]) + 1:]
    # passage = general_detokenize(pre + " *{}*".format(doc["challenge_text"]) + post)
    # noun = doc["response_text"]
    # pronoun = doc["challenge_text"]
    question = doc["fråga"]
    index=1
    context = ""
    while ("premiss_"+str(index) in doc):
        if (doc["premiss_"+str(index)]!= None) :
            context += "Premiss " +str(index) +": "+doc["premiss_"+str(index)]+"\n"
        index+=1
    text = (
            f"Passage:\n{context}"
            + f"Fråga: {question}\n"
            + "Svar:"
        )
    return text

   def doc_to_text(self, doc):
       return doc["query"]


