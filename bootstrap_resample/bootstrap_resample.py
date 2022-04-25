import datasets
from random import randint

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
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            )
        )

    def _compute(self, predictions, references, num_resamples=1000):
        diff_scores = [prediction - ref for prediction, ref in zip(predictions, references)]
        not_better_runs = 0
        for i in range(num_resamples):
            resample = [diff_scores[randint(0, len(diff_scores) - 1)]
                        for j in range(len(diff_scores))]
            if sum(resample) <= 0.0:
                not_better_runs += 1
        return not_better_runs / num_resamples
