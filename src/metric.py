import re
import numpy as np
import string
from collections import Counter
from evaluate import load
import spacy

# We need the following corpus for
# python -m spacy download en_core_web_sm


class SQuAD_eval:
    """
    The SQuAD exact match and SQuAD f1 metric
    Reference: https://github.com/huggingface/datasets/blob/0e1c629cfb9f9ba124537ba294a0ec451584da5f/metrics/squad/evaluate.py

    Parameters
    ----------
    name: str
        Choices: ['squad-f1', 'squad-em', 'squad', 'squad-precision', 'squad-recall']
        'squad' will output both 'squad-f1' and 'squad-em'.

    """

    def __init__(self, name="squad"):
        self._metric_name = name
        self.spacy_nlp = spacy.load("en_core_web_sm")
        assert name in [
            "squad-f1",
            "squad-em",
            "squad",
            "squad-pm",
            "squad-precision",
            "squad-recall",
        ]

    def convert2base(self, text: str) -> str:
        return " ".join([t.lemma_ for t in self.spacy_nlp(text)])

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def replace_underscore_with_space(text):
            return text.replace("_", " ")

        def replace_hyphen_with_space(text):
            return text.replace("-", " ")

        def lower(text):
            return text.lower()

        res = []
        if "_" in s or "-" in s:
            new_ans1 = white_space_fix(
                remove_articles(
                    remove_punc(
                        replace_hyphen_with_space(
                            replace_underscore_with_space(lower(s))
                        )
                    )
                )
            )
            res.append(new_ans1)

        new_answer = white_space_fix(remove_articles(remove_punc((lower(s)))))
        res.append(new_answer)
        new_answer = self.convert2base(new_answer)
        if new_answer not in res:
            res.append(new_answer)
        return res

    def exact_match_score(self, prediction, ground_truth):
        return prediction == ground_truth

    def _token_overlap(self, prediction, ground_truth):
        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        return num_same, prediction_tokens, ground_truth_tokens

    def precision_score(self, prediction, ground_truth):
        """
        len(matched tokens) / len(prediction tokens)

        Parameters
        ----------
        prediction: str
        ground_truth: str

        Returns
        -------
        float
            Range from 0 to 1.
        """
        num_same, prediction_tokens, ground_truth_tokens = self._token_overlap(
            prediction, ground_truth
        )
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(prediction_tokens)
        return precision

    def recall_score(self, prediction, ground_truth):
        """
        len(matched tokens) / len(ground truth tokens)

        Parameters
        ----------
        prediction: str
        ground_truth: str

        Returns
        -------
        float
            Range from 0 to 1.
        """
        num_same, prediction_tokens, ground_truth_tokens = self._token_overlap(
            prediction, ground_truth
        )
        if num_same == 0:
            return 0.0
        recall = 1.0 * num_same / len(ground_truth_tokens)
        return recall

    def f1_score(self, prediction, ground_truth):
        """
        (2 * precision * recall) / (precision + recall)

        Parameters
        ----------
        prediction: str
        ground_truth: str

        Returns
        -------
        float
            Range from 0 to 1.
        """
        precision = self.precision_score(prediction, ground_truth)
        recall = self.recall_score(prediction, ground_truth)
        if precision == 0 or recall == 0:
            return 0.0
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def metric_max_over_predictions_ground_truths(
        self, metric_fn, predictions, ground_truths
    ):
        """
        Return the maximum score over all the variant pairs

        Parameters
        ----------
        metric_fn: function
        predictions: list
        ground_truths: list

        Returns
        -------
        scalar: float
        """
        scores_for_ground_truths = []
        for prediction in predictions:
            for ground_truth in ground_truths:
                scores = metric_fn(prediction, ground_truth)
                scores_for_ground_truths.append(scores)
        return max(scores_for_ground_truths)

    def compute(self, predictions, references, level="all"):
        """

        Parameters
        ----------
        predictions: list
        references: list
        level: str
            To return averaged/individual metric scores
            Choices: ['all', 'individual']
                'all': the average score for all the predictions
                'individual': a list of scores for each individual (prediction, reference) pair

        Returns
        -------
        dict
        """
        f1, exact_match, precision, recall, pm = 0, 0, 0, 0, 0
        f1_l, exact_match_l, precision_l, recall_l = [], [], [], []
        for pred, ref in zip(predictions, references):
            if isinstance(ref, str):
                ref = self.normalize_answer(ref)
            else:
                # TODO we will support multi groundtruth in the future
                raise NotImplementedError

            if isinstance(pred, str):
                pred = self.normalize_answer(pred)
            else:
                # TODO we will support multi groundtruth in the future
                raise NotImplementedError

            exact_match_l.append(
                self.metric_max_over_predictions_ground_truths(
                    self.exact_match_score, pred, ref
                )
            )
            f1_l.append(
                self.metric_max_over_predictions_ground_truths(self.f1_score, pred, ref)
            )
            precision_l.append(
                self.metric_max_over_predictions_ground_truths(
                    self.precision_score, pred, ref
                )
            )
            recall_l.append(
                self.metric_max_over_predictions_ground_truths(
                    self.recall_score, pred, ref
                )
            )

        if len(predictions) != 0 and len(references) != 0:
            pm = np.sum(np.array(f1_l) != 0.0) / len(f1_l)
            exact_match = np.mean(exact_match_l)
            f1 = np.mean(f1_l)
            precision = np.mean(precision_l)
            recall = np.mean(recall_l)
        exact_match_l = np.array(exact_match_l).astype(int).tolist()
        assert (
            len(predictions)
            == len(references)
            == len(f1_l)
            == len(exact_match_l)
            == len(precision_l)
            == len(recall_l)
        )

        if self._metric_name == "squad-f1":
            res = {self._metric_name: f1}
            if level == "individual":
                res[f"individual_{self._metric_name}"] = f1_l
        elif self._metric_name == "squad-em":
            res = {self._metric_name: exact_match}
            if level == "individual":
                res[f"individual_{self._metric_name}"] = exact_match_l
        elif self._metric_name == "squad-precision":
            res = {self._metric_name: precision}
            if level == "individual":
                res[f"individual_{self._metric_name}"] = precision_l
        elif self._metric_name == "squad-recall":
            res = {self._metric_name: recall}
            if level == "individual":
                res[f"individual_{self._metric_name}"] = recall_l
        elif self._metric_name == "squad-pm":
            res = {self._metric_name: pm}
            if level == "individual":
                res[f"individual_{self._metric_name}"] = list(
                    (np.array(f1_l) != 0).astype(int)
                )

        else:
            res = {"squad-em": exact_match, "squad-f1": f1, "squad-pm": pm}
            if level == "individual":
                res["individual_squad-em"] = exact_match_l
                res["individual_squad-f1"] = f1_l
        return res


class BNGMetrics:
    """
    The class for multiple supported BNG metrics
    Now we support ['squad-em', 'squad-f1', 'squad', 'perplexity', 'bertscore-f1',
    'bertscore-precision', 'bertscore-recall', 'bertscore']
    as listed in `supported_metric_list`.

    Parameters
        ----------
        metric_names: str or list
            A metric name or a list of metric names
        device: str
            On which the contextual embedding model of bertscore and perplexity will be allocated on.
            If this argument is None, the model lives on cuda:0 if cuda is available.
        batch_size: int
            The batch size to run texts through the model.
            Used as the `batch_size` parameter in bertscore and perplexity APIs

    """

    def __init__(self, metric_names, device="cpu", batch_size=128):
        self.metric_names = metric_names
        self.device = device
        self.batch_size = batch_size
        self.predictions, self.references = [], []
        self.metrics_dict = self.load_evaluators()

    @property
    def supported_metric_list(self):
        return [
            "squad-em",
            "squad-f1",
            "squad",
            "squad-pm",
            "squad-precision",
            "squad-recall",
            "bertscore-f1",
            "bertscore-precision",
            "bertscore-recall",
            "bertscore",
        ]

    @property
    def num_pred(self):
        return len(self.predictions)

    @property
    def num_ref(self):
        return len(self.references)

    def load_evaluators(self):
        if isinstance(self.metric_names, str):
            return {self.metric_names: self._load_evaluator(self.metric_names)}
        elif isinstance(self.metric_names, list):
            return {
                metric_name: self._load_evaluator(metric_name)
                for metric_name in self.metric_names
            }
        else:
            raise NotImplementedError(
                "The metric_names are not a string or a list of strings. "
                "Please reset the metric_names of the BNGMetrics()."
            )

    def _load_evaluator(self, _metric_name):
        assert _metric_name in self.supported_metric_list
        if _metric_name in [
            "squad-em",
            "squad-f1",
            "squad",
            "squad-pm",
            "squad-precision",
            "squad-recall",
        ]:
            return SQuAD_eval(_metric_name)
        elif _metric_name in [
            "bertscore-f1",
            "bertscore-precision",
            "bertscore-recall",
            "bertscore",
        ]:
            return load("bertscore")
        else:
            return load(_metric_name)

    def add(self, prediction=None, reference=None):
        """Hold the same API as in huggingface evaluate
        https://github.com/huggingface/evaluate/blob/v0.1.2/src/evaluate/module.py#L505
        But we only accept the text prediction and reference

        Parameters
        ----------
        prediction: list
        reference: list

        Returns
        -------

        """
        assert isinstance(prediction, list) and isinstance(reference, list)
        self.predictions += prediction
        self.references += reference

    def clear(self):
        self.predictions, self.references = [], []

    def _compute_score(self, metric_name, predictions, references, level="all"):
        """

        Parameters
        ----------
        metric_name: str
            A metric name as listed in `supported_metric_list'
        predictions: list
            A list of string predictions
        references: list
            A list of string reference
        level: str
            Different metric computation levels for prediction and reference pairs.
            Choices: ['all', 'individual']
            Default: 'all'
            Note: only 'squad*' metrics support 'individual' level, which will return the metric score
                for each individual (prediction, reference) pair.

        Returns
        -------
        Dict
            Key: metric_name
            Value: average score for `all'/a list of scores for `individual'
        """

        if metric_name in [
            "squad-em",
            "squad-f1",
            "squad",
            "squad-pm",
            "squad-precision",
            "squad-recall",
        ]:
            return self.metrics_dict[metric_name].compute(
                predictions=predictions, references=references, level=level
            )
        elif metric_name in [
            "bertscore-f1",
            "bertscore-precision",
            "bertscore-recall",
            "bertscore",
        ]:
            results = self.metrics_dict[metric_name].compute(
                predictions=[ele.lower().strip() for ele in predictions],
                references=[ele.lower().strip() for ele in references],
                lang="en",
                device=self.device,
                batch_size=self.batch_size,
                rescale_with_baseline=True,
            )

            if metric_name == "bertscore-f1":
                ret = {metric_name: sum(results["f1"]) / len(results["f1"])}
                if level == "individual":
                    ret[f"individual_{metric_name}"] = results["f1"]
            elif metric_name == "bertscore-precision":
                ret = {
                    metric_name: sum(results["precision"]) / len(results["precision"])
                }
                if level == "individual":
                    ret[f"individual_{metric_name}"] = results["precision"]
            elif metric_name == "bertscore-recall":
                ret = {metric_name: sum(results["recall"]) / len(results["recall"])}
                if level == "individual":
                    ret[f"individual_{metric_name}"] = results["recall"]
            elif metric_name == "bertscore":
                key_names = ["precision", "recall", "f1"]
                ret = {
                    f"bertscore-{key_name}": sum(results[key_name])
                    / len(results[key_name])
                    for key_name in key_names
                }
                if level == "individual":
                    for key_name in key_names:
                        ret[f"individual_bertscore-{key_name}"] = results[key_name]
            return ret
        else:
            raise NotImplementedError("Undefined metric name: {}".format(metric_name))

    def compute_scores(self, predictions=None, references=None, level="all"):
        if predictions is None and references is None:
            assert self.num_pred > 0
            preds, refs = self.predictions, self.references
        else:
            preds, refs = predictions, references
        all_metric_values = {}
        for metric_name in self.metrics_dict.keys():
            res_dict = self._compute_score(
                metric_name=metric_name, predictions=preds, references=refs, level=level
            )
            all_metric_values = {**all_metric_values, **res_dict}
        return all_metric_values
