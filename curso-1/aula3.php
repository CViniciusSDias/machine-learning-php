<?php

declare(strict_types=1);

use Rubix\ML\Classifiers\{SVC};
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\ZScaleStandardizer;

require __DIR__ . '/../vendor/autoload.php';

$csvUrlStream = fopen('https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv', 'r');
$headers = fgets($csvUrlStream);

$data = [];
while (!feof($csvUrlStream)) {
    $line = fgetcsv($csvUrlStream);
    $line[3] = $line[0] === '1' ? 'N Finaliza' : 'Finaliza';

    $data[] = $line;
}

$dataset = new Labeled(array_map(fn (array $line): array => [intval($line[1]), intval($line[2])], $data), array_column($data, 3));
$dataset->apply(new ZScaleStandardizer());

[$training, $testing] = $dataset->stratifiedSplit(0.75);

$estimator = new SVC();
$estimator->train($training);

$predictions = $estimator->predict($testing);

$metric = new Accuracy();
$accuracyPercent = $metric->score($predictions, $testing->labels()) * 100;

echo "Treinamos com {$training->count()} elementos e testamos com {$testing->count()} elementos." . PHP_EOL;
echo "A precisão foi de: {$accuracyPercent}%" . PHP_EOL;

$my_predictions = array_fill(0, $testing->count(), 'Finaliza');
echo round($metric->score($my_predictions, $testing->labels()) * 100, 2) . '%' . PHP_EOL;