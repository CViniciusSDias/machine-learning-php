<?php

declare(strict_types=1);

use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;

require __DIR__ . '/../vendor/autoload.php';

$dataset = Labeled::fromIterator(new CSV(__DIR__ . '/customer-churn.csv', header: true));

[$training, $testing] = $dataset->stratifiedSplit(0.7);

$classificationTree = new ClassificationTree();
$classificationTree->train($training);

$predictions = $classificationTree->predict($testing);
var_dump($classificationTree->featureImportances());

$accuracy = new Accuracy();
$score = $accuracy->score($predictions, $testing->labels());

echo $score . PHP_EOL;