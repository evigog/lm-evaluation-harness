from . common import HFTask, yesno
from ..utils import general_detokenize
from ..metrics import mean, acc_all, metric_max_over_ground_truths
import os
from lm_eval.base import rf
from lm_eval.base import MultipleChoiceTask

class SweWsc(HFTask):
   VERSION = 0
   DATASET_PATH = "AI-Sweden/SuperLim"
   DATASET_NAME = "SweWsc"
   DATA_FILES = {"test":"SweWsc/test.csv"}
   USE_AUTH_TOKEN = os.environ['HF_TOKEN']

   def has_training_docs(self):
        return False

   def has_validation_docs(self):
        return False

   def has_test_docs(self):
        return True

   def fewshot_description(self):
    return "Does the noun match the pronoun?"

   def doc_to_text(self, doc):
    raw_passage = doc["text"]
    # NOTE: HuggingFace span indices are word-based not character-based.
    pre = " ".join(raw_passage.split()[:doc["span1_begin"]])
    post = raw_passage[len(pre) + len(doc["span1_text"]) + 1:]
    passage = general_detokenize(pre + " *{}*".format(doc["span1_text"]) + post)
    noun = doc["span2_text"]
    pronoun = doc["span1_text"]
    text = (
            f"Passage: {passage}\n"
            + f"Fråga: I avsnittet ovan, syftar pronomenet \"*{pronoun}*\" på \"*{noun}*\"?\n"
            + "Svar:"
        )
    return text

   def doc_to_target(self, doc):
    return " " + yesno(doc['label'])

   def construct_requests(self, doc, ctx):

        ll_yes, _ = rf.loglikelihood(ctx, ' yes')
        ll_no, _ = rf.loglikelihood(ctx, ' no')

        return ll_yes, ll_no

   def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc['responses'][0]["correct"]

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
   USE_AUTH_TOKEN = os.environ['HF_TOKEN']

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
