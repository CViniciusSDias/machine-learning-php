<?php

declare(strict_types=1);

use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;

require __DIR__ . '/../vendor/autoload.php';

$dataset = Labeled::fromIterator(new CSV(__DIR__ . '/customer-churn.csv', header: true))
    ->apply(new NumericStringConverter())
    ->apply(new LambdaFunction(function (array &$samples) {
        $simNao = fn (string $original) => match ($original) {
            'Sim' => 1,
            'Nao' => 0,
            default => $original,
        };

        $samples[1] = $simNao($samples[1]);
        $samples[2] = $simNao($samples[2]);
        $samples[4] = $simNao($samples[4]);
        $samples[14] = $simNao($samples[14]);
    }))
    ->apply(new OneHotEncoder())
    ->apply(new ZScaleStandardizer());

[$training, $testing] = $dataset->stratifiedSplit(0.7);

$knn = new KNearestNeighbors(kernel: new Euclidean());
$knn->train($training);

$predictions = $knn->predict($testing);

$accuracy = new Accuracy();
$score = $accuracy->score($predictions, $testing->labels());

echo $score . PHP_EOL;