using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Threading.Tasks;

namespace MLTest.Consoles
{
    class Program
    {
        const string _dataPath = @"C:\Users\jmesa\Desktop\ML.Net\MLTest\MLTest.Console\Data\Data.txt";
        const string _modelPath = @"C:\Users\jmesa\Desktop\ML.Net\MLTest\MLTest.Console\Data\Model.zip";

        static void YMain(string[] args)
        {
            Task.Run(() =>
            {
                PredictionModel<Abalone, AbaloneAgePrediction> model = Train().Result;
                Abalone abalone1 = new Abalone()
                {
                    Lenght = 0.524f,
                    Diameter = 0.408f,
                    Height = 0.140f,
                    WholeWeight = 0.829f,
                    ShellWeight = 0.239f,
                    Age = 0,
                };
                var prediction = model.Predict(abalone1);
                Console.WriteLine("Predicted Age is {0}", Math.Floor(prediction.Age));
            });
            Console.ReadLine();
        }

        static async Task Main(string[] args)
        {
            PredictionModel<Abalone, AbaloneAgePrediction> model = await Train();
            Abalone abalone1 = new Abalone
            {
                Lenght = 0.524f,
                Diameter = 0.408f,
                Height = 0.140f,
                WholeWeight = 0.829f,
                ShellWeight = 0.239f,
                Age = 0,
            };
            Abalone abalone2 = new Abalone
            {
                Lenght = 0.455f,
                Diameter = 0.365f,
                Height = 0.095f,
                WholeWeight = 0.514f,
                ShellWeight = 0.15f,
                Age = 0,
            };

            var prediction = model.Predict(abalone1);
            Console.WriteLine(string.Format("Predicted Age is {0}", Math.Floor(prediction.Age)));
            prediction = model.Predict(abalone2);
            Console.WriteLine(string.Format("Predicted Age is {0}", Math.Floor(prediction.Age)));
            Console.ReadLine();
        }

        public static async Task<PredictionModel<Abalone, AbaloneAgePrediction>> Train()
        {

            var pipeLine = new LearningPipeline();
            //{
            //    new TextLoader(_dataPath).CreateFrom<Abalone>(separator: ','),
            //    new ColumnCopier(("Age", "Label")),
            //    new CategoricalOneHotVectorizer("Sex"),
            //    new ColumnConcatenator("Features","Sex","Lenght","Diameter","Height","WholeWeight","ShellWeigth"),
            //    new FastTreeRegressor()
            //};

            pipeLine.Add(new TextLoader(_dataPath).CreateFrom<Abalone>(separator: ','));
            pipeLine.Add(new ColumnCopier(("Age", "Label")));
            pipeLine.Add(new CategoricalOneHotVectorizer("Sex"));
            pipeLine.Add(new ColumnConcatenator("Features", "Sex", "Lenght", "Diameter", "Height", "WholeWeight", "ShellWeight"));
            pipeLine.Add(new FastTreeRegressor());

            PredictionModel<Abalone, AbaloneAgePrediction> model = pipeLine.Train<Abalone, AbaloneAgePrediction>();
            await model.WriteAsync(_modelPath);
            return model;


        }
    }
}
