<?php

declare(strict_types=1);

require __DIR__ . '/../vendor/autoload.php';

use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Classifiers\SVC;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Kernels\SVM\Linear;

$csvUrlStream = fopen('https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv', 'r');
$headers = fgets($csvUrlStream);

$samples = [];
$labels = [];
while (!feof($csvUrlStream)) {
    $line = fgetcsv($csvUrlStream);
    $labels[] = $line[3] === '0' ? 'Não Compraria' : 'Compraria';

    $samples[] = [$line[0], $line[1], $line[2]];
}

$dataset = new Labeled($samples, $labels);

[$training, $testing] = $dataset->stratifiedSplit(0.75);

// $estimator = new SVC(kernel: new Linear());
$estimator = new NaiveBayes();
$estimator->train($training);

$predictions = $estimator->predict($testing);

$metric = new Accuracy();
$score = $metric->score($predictions, $testing->labels());

echo $score;