using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Runtime.Serialization;
using System.ServiceModel;
using System.ServiceModel.Activation;
using System.ServiceModel.Web;
using System.Text;
using System.Web.Http;
//using TonalityClassifier;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using System.Text.RegularExpressions;

namespace ClassificationWebService
{
    [ServiceContract(Namespace = "")]
    [AspNetCompatibilityRequirements(RequirementsMode = AspNetCompatibilityRequirementsMode.Allowed)]
    public class ClassificationService
    {
        // Чтобы использовать протокол HTTP GET, добавьте атрибут [WebGet]. (По умолчанию ResponseFormat имеет значение WebMessageFormat.Json.)
        // Чтобы создать операцию, возвращающую XML,
        //     добавьте [WebGet(ResponseFormat=WebMessageFormat.Xml)]
        //     и включите следующую строку в текст операции:
        //         WebOperationContext.Current.OutgoingResponse.ContentType = "text/xml";
        // Добавьте здесь дополнительные операции и отметьте их атрибутом [OperationContract]
        [OperationContract]
        public void DoWork()
        {
            // Добавьте здесь реализацию операции
            return;
        }

        public class SentimentData
        {
            [LoadColumn(0)]
            public string SentimentText;

            [LoadColumn(1), ColumnName("Label")]
            public bool Sentiment;
        }

        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Prediction { get; set; }

            public float Probability { get; set; }

            public float Score { get; set; }
        }

        private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "rus_text.txt");
        public static void GetTonalityClass(string text)
        {
            MLContext mlContext = new MLContext();

            //загружаем корпус текстов, получаем наборы для обучения и для тестирования:
            TrainCatalogBase.TrainTestData splitDataView = LoadData(mlContext);

            //преобразовывем данные из файла и обучаяем нейросеть:
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            //оценим созданную нейросеть (получим метрики):
            //Evaluate(mlContext, model, splitDataView.TestSet);
            //воспользуемся моделью для классификации тональности обращения:
            //string tonclass = UseModel(text, mlContext, model);
            //return tonclass;

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

        private static List<string> lst = new List<string>
        {
            "Arrays",
            "Queues",
            "Stacks"
        };

        [WebGet(UriTemplate = "/Element")]
        public string GetAllElement() => String.Join(",", lst);
        /*{
            System.Net.WebRequest req = System.Net.WebRequest.Create("http://localhost:60051/ClassificationService.svc/Element");
            req.Method = "POST";
            req.Timeout = 100000;
            req.ContentType = "application/json";
            byte[] sentData = Encoding.GetEncoding(1251).GetBytes("str=Отличная работа");
            req.ContentLength = sentData.Length;
            System.IO.Stream sendStream = req.GetRequestStream();
            sendStream.Write(sentData, 0, sentData.Length);
            sendStream.Close();
            //System.Net.WebResponse res = req.GetResponse();
        }*/

        [WebGet(UriTemplate = "/Element/{ElementId}")]
        public string GetElementByID(string ElementId)
        {
            int pid;
            if (!Int32.TryParse(ElementId, out pid))
            {
                throw new HttpResponseException(HttpStatusCode.BadRequest);
            }
            return lst[pid];
        }

        [WebInvoke(Method = "POST", RequestFormat = WebMessageFormat.Json,
            UriTemplate = "/Element", ResponseFormat = WebMessageFormat.Json,
            BodyStyle = WebMessageBodyStyle.Wrapped)]
        public void AddElement(string str) => GetTonalityClass(str);//lst.Add(TonalClassifier.GetTonalityClass(str));

        [WebInvoke(Method = "DELETE", RequestFormat = WebMessageFormat.Json,
            UriTemplate = "/Element/{ElementId}", ResponseFormat = WebMessageFormat.Json,
            BodyStyle = WebMessageBodyStyle.Wrapped)]
        public void DeleteElement(string ElementId)
        {

            int pid;
            if (!Int32.TryParse(ElementId, out pid))
            {
                throw new HttpResponseException(HttpStatusCode.BadRequest);
            }
            lst.RemoveAt(pid);
        }
    }
}
