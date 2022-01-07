"""
To-do:
    - WSC requires free-form generation
    - ReCoRD
"""
import numpy as np
import sklearn
import transformers.data.metrics.squad_metrics as squad_metrics
from . common import HFTask, yesno, janej
from lm_eval.base import rf
from ..metrics import mean, acc_all, metric_max_over_ground_truths
from ..utils import general_detokenize
import os


class BoolQ(HFTask): #Test has only -1, something wrong with the labels
    VERSION = 0
    DATASET_PATH = "AI-Sweden/super_glue_sv"
    DATASET_NAME = "boolq"
    DATA_FILES = {"train":"boolq/train.csv","validation":"boolq/val.csv"}
    USE_AUTH_TOKEN = os.environ['HF_TOKEN']

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out actual description
        return "Läs följande avsnitt och besvara varje fråga med ett ja eller ett nej."

    def doc_to_text(self, doc):
        return f"{doc['passage']}\nFråga: {doc['question']}\nSvar:"
    
    def doc_to_target(self, doc):
        return " " + janej(doc['label']==1)

    def construct_requests(self, doc, ctx):

        ll_yes, _ = rf.loglikelihood(ctx, ' Ja')
        ll_no, _ = rf.loglikelihood(ctx, ' Nej')

        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]

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


class CommitmentBank(HFTask):
    VERSION = 0
    DATASET_PATH = "AI-Sweden/super_glue_sv"
    DATASET_NAME = "cb"
    DATA_FILES = {"train": "cb/train.csv", "validation": "cb/val.csv"}
    USE_AUTH_TOKEN = os.environ['HF_TOKEN']

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out actual description
        return """Med tanke på en premiss och en hypotes, klassificera om författaren till premisserna är övertygad om att hypotesen är sann.De tre möjliga etiketterna är sant, falskt eller varken eller."""

    def doc_to_text(self, doc):
        return "{}\nFråga: {}. Sant, Falskt eller Ingetdera?\nSvar:".format(
            doc["premise"],
            doc["hypothesis"],
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({"entailment": "Sant", "neutral": "Ingetdera", "contradiction": "Falskt"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, ' Sant')
        ll_neither, _ = rf.loglikelihood(ctx, ' Ingetdera')
        ll_false, _ = rf.loglikelihood(ctx, ' Falskt')

        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        gold = {"entailment": 0, "neutral": 1, "contradiction": 2}[doc["label"]]
        pred = np.argmax(results)
        acc = 1. if pred == gold else 0.

        return {
            "acc": acc,
            "f1": (pred, gold)
        }
    
    def higher_is_better(self):
        return {
            "acc": True,
            "f1": True
        }

    @classmethod
    def cb_multi_fi(cls, items):
        preds, golds = zip(*items)
        preds = np.array(preds)
        golds = np.array(golds)
        f11 = sklearn.metrics.f1_score(y_true=golds == 0, y_pred=preds == 0)
        f12 = sklearn.metrics.f1_score(y_true=golds == 1, y_pred=preds == 1)
        f13 = sklearn.metrics.f1_score(y_true=golds == 2, y_pred=preds == 2)
        avg_f1 = mean([f11, f12, f13])
        return avg_f1
    
    def aggregation(self):
        return {
            "acc": mean,
            "f1": self.cb_multi_fi,
        }


class Copa(HFTask):
    VERSION = 0
    DATASET_PATH = "AI-Sweden/super_glue_sv"
    DATASET_NAME = "copa"
    DATA_FILES = {"train": "copa/train.csv", "validation": "copa/val.csv"}
    USE_AUTH_TOKEN = os.environ['HF_TOKEN']

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out actual description
        return "Om du har en premiss och ett alternativ som har ett orsakssamband med premiss och ett annat som inte har det, välj då det mest trovärdiga alternativet."

    def doc_to_text(self, doc): #I feel that this could be implemented with a better prompt
        # Drop the period
        connector = {
            "cause": "eftersom",
            "effect": "därför",
        }[doc["question"]]
        return doc["premise"].strip()[:-1] + f" {connector}"

    def doc_to_target(self, doc):
        correct_choice = doc["choice1"] if doc["label"] == 0 else doc["choice2"]
        # Connect the sentences
        return " " + self.convert_choice(correct_choice)

    def construct_requests(self, doc, ctx):
        choice1 = " " + self.convert_choice(doc["choice1"])
        choice2 = " " + self.convert_choice(doc["choice2"])
        
        ll_choice1, _ = rf.loglikelihood(ctx, choice1)
        ll_choice2, _ = rf.loglikelihood(ctx, choice2)

        return ll_choice1, ll_choice2

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        acc = 1. if pred == gold else 0.

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

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]

#not in data:
#TODO MultiRC(HFTask):
#TODO ReCoRD(HFTask):
#TODO WiC(HFTask):
#TODO WSC(HFTask):
#TODO ReCoRD(HFTask):
#not in eleuther:
#TODO rte(HFTask):
#not in both:
#TODO AX-b(HFTask):
#TODO AX-g(HFTask):