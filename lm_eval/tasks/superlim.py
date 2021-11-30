from . common import HFTask, yesno
from ..utils import general_detokenize
from ..metrics import mean, acc_all, metric_max_over_ground_truths
import os
from lm_eval.base import rf

class SweWsc(HFTask):
   VERSION = 0
   DATASET_PATH = "AI-Sweden/SuperLim"
   DATASET_NAME = "SweWsc"
   DATA_FILES = {"test":"SweWsc/test.jsonl"}
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
    pre = " ".join(raw_passage.split()[:doc["challenge"]["location"]["start"]])
    post = raw_passage[len(pre) + len(doc["challenge"]["text"]) + 1:]
    passage = general_detokenize(pre + " *{}*".format(doc['challenge']["text"]) + post)
    noun = doc["responses"][0]["text"] #todo add more responses. now we consider only the first
    pronoun = doc["challenge"]["text"]
    text = (
            f"Passage: {passage}\n"
            + f"Fråga: I avsnittet ovan, syftar pronomenet \"*{pronoun}*\" på \"*{noun}*\"?\n"
            + "Svar:"
        )
    return text

   def doc_to_target(self, doc):
    return " " + yesno(doc['responses'][0]["correct"])

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


