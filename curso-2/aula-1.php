<?php

declare(strict_types=1);

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\OneHotEncoder;

require __DIR__ . '/../vendor/autoload.php';

$dataset = Unlabeled::fromIterator(new CSV(__DIR__ . '/customer-churn.csv', header: true))
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
    ->apply(new OneHotEncoder());

var_dump($dataset->shape());
