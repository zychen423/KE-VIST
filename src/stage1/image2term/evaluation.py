from nlgeval import compute_metrics
metrics_dict = compute_metrics(hypothesis='./test/pred.txt',
                                       references=['./test/gt.txt'])
