using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using System.Text.RegularExpressions;

namespace TonalityClassifier
{
    public class TonalClassifier
    {
        private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "rus_text.txt");
 
        public static string GetTonalityClass(string text)
        {
            MLContext mlContext = new MLContext();

            //загружаем корпус текстов, получаем наборы для обучения и для тестирования:
            TrainCatalogBase.TrainTestData splitDataView = LoadData(mlContext);

            //преобразовывем данные из файла и обучаяем нейросеть:
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            //оценим созданную нейросеть (получим метрики):
            //Evaluate(mlContext, model, splitDataView.TestSet);
            //воспользуемся моделью для классификации тональности обращения:
             string tonclass =UseModel(text,mlContext, model);
             return tonclass;

        }

        protected static TrainCatalogBase.TrainTestData LoadData(MLContext mlContext)
        {//ф-я загрузки корпуса текстов, разделения корпуса на наборы данных для обучения и тестирования
            Microsoft.Data.DataView.IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainCatalogBase.TrainTestData splitDataView = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);

            return splitDataView;
        }

        protected static ITransformer BuildAndTrainModel(MLContext mlContext, Microsoft.Data.DataView.IDataView splitTrainSet)
        {//ф-я преобразования данных, обучения нейросети, прогнозирования тональности на основе тестовых данных
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
            .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));
            //выбрали алгоритм обучения по дереву принятия решений

            var model = estimator.Fit(splitTrainSet);//обучаем сеть
            return model;
        }

        protected static void Evaluate(MLContext mlContext, ITransformer model, Microsoft.Data.DataView.IDataView splitTestSet)
        {//метод загрузки тестового набора данных и оценки нейросети,создание метрик
            Microsoft.Data.DataView.IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Показатели качества сети:");
            Console.WriteLine($"Точность: {metrics.Accuracy:P2}");
            Console.WriteLine($"Баллы: {metrics.F1Score:P2}");
        }

        protected static string UseModel(string text, MLContext mlContext, ITransformer model)
        {//тестируем модель
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            SentimentData sampleStatement = new SentimentData//загружаем наш текст
            {
                SentimentText = text
            };
            var resultprediction = predictionFunction.Predict(sampleStatement);
            string tonality_class = (resultprediction.Score > 3) ? "Положительный" : (resultprediction.Score < -3) ? "Отрицательный" : "Нейтральный";

            //Console.WriteLine($"Тестовый текст: {sampleStatement.SentimentText} | Класс тональности:{tonality_class} | Значение: {resultprediction.Score}");
            //Console.WriteLine();
            //возвращаем класс тональности исходного текста:
            return tonality_class;
        }

    }
}

