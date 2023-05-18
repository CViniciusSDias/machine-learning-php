<?php

declare(strict_types=1);

use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;

require 'vendor/autoload.php';

$csvUrlStream = fopen('https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv', 'r');

// Ignora cabeçalho
$headers = fgetcsv($csvUrlStream);
unset($headers);

$samples = [];
$labels = [];
while (!feof($csvUrlStream)) {
    [$id, $milhas_por_ano, $ano_modelo, $preco, $vendido] = fgetcsv($csvUrlStream);

    $samples[] = [
        $milhas_por_ano * 1.60934,
        date('Y') - $ano_modelo,
        floatval($preco),
    ];
    $labels[] = $vendido;
}
fclose($csvUrlStream);

$dataset = new Labeled($samples, $labels);
[$training, $testing] = $dataset->stratifiedSplit(0.75);

$estimator = new ClassificationTree();
$estimator->train($training);

$predictions = $estimator->predict($testing);

$metric = new Accuracy();
echo $metric->score($predictions, $testing->labels()) . PHP_EOL;
