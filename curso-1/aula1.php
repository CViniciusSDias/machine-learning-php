<?php

use Rubix\ML\Classifiers\SVC;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Kernels\SVM\Linear;

require __DIR__ . '/../vendor/autoload.php';

$porco1 = [0, 1, 0];
$porco2 = [0, 1, 1];
$porco3 = [1, 1, 0];
$cachorro1 = [0, 1, 1];
$cachorro2 = [1, 0, 1];
$cachorro3 = [1, 1, 1];

$dados = [$porco1, $porco2, $porco3, $cachorro1, $cachorro2, $cachorro3];
$classes = ['porco', 'porco', 'porco', 'cachorro', 'cachorro', 'cachorro'];

$dataset = new Labeled($dados, $classes);

$estimador = new SVC(kernel: new Linear());
$estimador->train($dataset);

$dataset = new Unlabeled([
    [1, 1, 1],
    [1, 1, 0],
    [0, 1, 1],
]);
$previsoes = $estimador->predict($dataset);

var_dump($previsoes);