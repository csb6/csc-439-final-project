import datasets
from random import randint

# Implements non-parametric bootstrap resampling in the Huggingface Datasets framework
class bootstrap_resample(datasets.Metric):
    # Code for info. function based on:
    #  https://github.com/huggingface/datasets/blob/master/metrics/f1/f1.py
    def _info(self):
        return datasets.MetricInfo(
            description="Non-parametric bootstrap resampling",
            citation=None,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                    "true_labels": datasets.Sequence(datasets.Value("int32"))
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                    "true_labels": datasets.Value("int32")
                }
            )
        )

    def _compute(self, predictions, references, true_labels, num_resamples=1000):
        diff_scores = []
        for prediction, ref, true_label in zip(predictions, references, true_labels):
            if prediction == true_label and ref == true_label:
                # Both correct
                diff_scores.append(0)
            elif prediction == true_label and ref != true_label:
                # Experience helps
                diff_scores.append(1)
            elif prediction != true_label and ref == true_label:
                # Experience hurts
                diff_scores.append(-1)
            else:
                # Both incorrect
                diff_scores.append(0)
        not_better_runs = 0
        for i in range(num_resamples):
            resample = [diff_scores[randint(0, len(diff_scores) - 1)]
                        for j in range(len(diff_scores))]
            if sum(resample) <= 0:
                not_better_runs += 1
        return not_better_runs / num_resamples
