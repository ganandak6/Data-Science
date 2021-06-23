using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace ML_Heart_Desease
{    
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("ML Process started...");
            #region 1. Create ML.NET context
            var mlContext = new MLContext();
            #endregion

            #region 2. Load Data
            IDataView data = mlContext.Data.LoadFromTextFile<HeartDesease>(@"..\..\..\Data\heart-disease.csv", ',',true);
            #endregion
            
            #region Split data for training and testing
            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            IDataView trainingData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;

            // Preview the data. 
            var dataPreview = data.Preview();

            Console.WriteLine($"There are {trainingData.Schema.Count} columns" + $" in the training dataset.");
            Console.WriteLine($"There are {testData.Schema.Count} columns" +  $" in the test dataset.");
            #endregion
            #region 3. Get the column name of input features.

            string[] featureColumnNames =
                data.Schema
                    .Select(column => column.Name)
                    .Where(columnName => columnName != "Label").ToArray();
            #endregion

            #region 4. Define estimator with data pre-processing steps
            IEstimator<ITransformer> dataPrepareEstimator = mlContext.Transforms.Concatenate("Features", featureColumnNames)
                                                            .Append(mlContext.Transforms.NormalizeMinMax("Features"))                                                            
                                                            .Append(mlContext.BinaryClassification.Trainers.AveragedPerceptron());
            #endregion

            #region Train the model

            ITransformer model = dataPrepareEstimator.Fit(trainingData);      
            // Use trained model to make inferences on test data
            IDataView testDataPredictions = model.Transform(testData);
            var calibratorEstimator = mlContext.BinaryClassification.Calibrators.Platt();
            ITransformer calibratorTransformer = calibratorEstimator.Fit(testDataPredictions);
            var finalData = calibratorTransformer.Transform(testDataPredictions).Preview();
            PrintRowViewValues(finalData);
            #endregion

            #region Evaluate Model
            CalibratedBinaryClassificationMetrics trainedModelMetrics = mlContext.BinaryClassification.Evaluate(calibratorTransformer.Transform(testDataPredictions));
            // Print out accuracy metric
            Console.WriteLine("Accuracy " + trainedModelMetrics.Accuracy);
            #endregion
        }

        private static void PrintRowViewValues(Microsoft.ML.Data.DataDebuggerPreview data)
        {
            var firstRows = data.RowView.Take(5);

            foreach (Microsoft.ML.Data.DataDebuggerPreview.RowInfo row in firstRows)
            {
                foreach (var kvPair in row.Values)
                {
                    if (kvPair.Key.Equals("Score") || kvPair.Key.Equals("Probability"))
                        Console.Write($" {kvPair.Key} {kvPair.Value} ");
                }
                Console.WriteLine();
            }
        }
    }
}
