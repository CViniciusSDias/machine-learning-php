<?php

declare(strict_types=1);

use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Kernels\Distance\Cosine;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\ML\Report;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;

require __DIR__ . '/../vendor/autoload.php';

$dataset = Labeled::fromIterator(new CSV(__DIR__ . '/customer-churn.csv', header: true));
[$training, $testing] = $dataset->stratifiedSplit(0.8);

$transformedDataset = $dataset
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

[$transformedTraining, $transformedTesting] = $transformedDataset->stratifiedSplit(0.8);

$knnEuclidean = new KNearestNeighbors(kernel: new Euclidean());
$knnEuclidean->train($transformedTraining);
$predictionsFromKnnEuclidean = $knnEuclidean->predict($transformedTesting);

$knnCosine = new KNearestNeighbors(kernel: new Cosine());
$knnCosine->train($transformedTraining);
$predictionsFromKnnCosine = $knnCosine->predict($transformedTesting);

$knnManhattan = new KNearestNeighbors(kernel: new Manhattan());
$knnManhattan->train($transformedTraining);
$predictionsFromKnnManhattan = $knnManhattan->predict($transformedTesting);

$knnMinkowski = new KNearestNeighbors(kernel: new Minkowski());
$knnMinkowski->train($transformedTraining);
$predictionsFromKnnMinkowski = $knnMinkowski->predict($transformedTesting);

$naiveBayes = new NaiveBayes();
$naiveBayes->train($training);
$predictionsFromNb = $naiveBayes->predict($testing);

$classificationTree = new ClassificationTree();
$classificationTree->train($training);
$predictionsFromTree = $classificationTree->predict($testing);

$confusionMatrix = new ConfusionMatrix();

$confusionMatrixKnnEuclidean = $confusionMatrix->generate($predictionsFromKnnEuclidean, $transformedTesting->labels());
$confusionMatrixKnnCosine = $confusionMatrix->generate($predictionsFromKnnCosine, $transformedTesting->labels());
$confusionMatrixKnnManhattan = $confusionMatrix->generate($predictionsFromKnnManhattan, $transformedTesting->labels());
$confusionMatrixKnnMinkowski = $confusionMatrix->generate($predictionsFromKnnMinkowski, $transformedTesting->labels());
$confusionMatrixNb = $confusionMatrix->generate($predictionsFromNb, $testing->labels());
$confusionMatrixTree = $confusionMatrix->generate($predictionsFromTree, $testing->labels());

$accuracy = new Accuracy();

function calculatePrecision(Report $confusionMatrix): float
{
    $truePositives = $confusionMatrix['Sim']['Sim'];
    $falsePositives = $confusionMatrix['Sim']['Nao'];

    return $truePositives / ($truePositives + $falsePositives);
}

function calculateRecall(Report $confusionMatrix): float
{
    $truePositives = $confusionMatrix['Sim']['Sim'];
    $falseNegatives = $confusionMatrix['Nao']['Sim'];

    return $truePositives / ($truePositives + $falseNegatives);
}

echo 'KNN Euclidean:' . PHP_EOL;
echo 'Accuracy: ' . $accuracy->score($predictionsFromKnnEuclidean, $transformedTesting->labels()) . PHP_EOL;
echo 'Precision: ' . calculatePrecision($confusionMatrixKnnEuclidean) . PHP_EOL;
echo 'Recall: ' . calculateRecall($confusionMatrixKnnEuclidean) . PHP_EOL;
echo PHP_EOL;

echo 'KNN Cosine:' . PHP_EOL;
echo 'Accuracy: ' . $accuracy->score($predictionsFromKnnCosine, $transformedTesting->labels()) . PHP_EOL;
echo 'Precision: ' . calculatePrecision($confusionMatrixKnnCosine) . PHP_EOL;
echo 'Recall: ' . calculateRecall($confusionMatrixKnnCosine) . PHP_EOL;
echo PHP_EOL;

echo 'KNN Manhattan:' . PHP_EOL;
echo 'Accuracy: ' . $accuracy->score($predictionsFromKnnManhattan, $transformedTesting->labels()) . PHP_EOL;
echo 'Precision: ' . calculatePrecision($confusionMatrixKnnManhattan) . PHP_EOL;
echo 'Recall: ' . calculateRecall($confusionMatrixKnnManhattan) . PHP_EOL;
echo PHP_EOL;

echo 'KNN Minkowski:' . PHP_EOL;
echo 'Accuracy: ' . $accuracy->score($predictionsFromKnnMinkowski, $transformedTesting->labels()) . PHP_EOL;
echo 'Precision: ' . calculatePrecision($confusionMatrixKnnMinkowski) . PHP_EOL;
echo 'Recall: ' . calculateRecall($confusionMatrixKnnMinkowski) . PHP_EOL;
echo PHP_EOL;

echo 'Naive Bayes:' . PHP_EOL;
echo 'Accuracy: ' . $accuracy->score($predictionsFromNb, $testing->labels()) . PHP_EOL;
echo 'Precision: ' . calculatePrecision($confusionMatrixNb) . PHP_EOL;
echo 'Recall: ' . calculateRecall($confusionMatrixNb) . PHP_EOL;
echo PHP_EOL;

echo 'Classification Tree:' . PHP_EOL;
echo 'Accuracy: ' . $accuracy->score($predictionsFromTree, $testing->labels()) . PHP_EOL;
echo 'Precision: ' . calculatePrecision($confusionMatrixTree) . PHP_EOL;
echo 'Recall: ' . calculateRecall($confusionMatrixTree) . PHP_EOL;