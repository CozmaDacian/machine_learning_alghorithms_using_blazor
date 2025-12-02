using System;


namespace Evaluation
{

    public interface IEvaluation
    {
        string Name { get; }
        double Calculate(double[] expected, double[] prediction);
        
    }


    public static class ConfusionMatrix
    {

        public static (int TP,int TN,int FP,int FN) CalculateConfusionForOne(double[] expected, double[] prediction,double targetClass)
        {
            int TP = 0, TN = 0, FP = 0, FN = 0;
            
            for (int i = 0; i < expected.Length; i++)
            {
                // Used for tolerance in case expected - prediction == 0.001
                bool isActualPos = Math.Abs(expected[i] - targetClass) < 0.1;
                bool isPredPos = Math.Abs(prediction[i] - targetClass) < 0.1;


                if (isActualPos && isPredPos) TP++;
                else if (!isActualPos && !isPredPos) TP++;
                else if (!isActualPos && isPredPos) FP++;
                else if (isActualPos && !isPredPos) FN++;
            }
            return (TP, TN, FP, FN);
        }
        public static List<double> GetUniqueClasses(double[] actual)
        {
            return actual.Distinct().OrderBy(x => x).ToList();
        }
   }

    public class Recall : IEvaluation
    {
        public string Name => "Recall";
        public double Calculate(double[] actual, double[] predicted)
        {
            var classes = ConfusionMatrix.GetUniqueClasses(actual);
            double totalRecall = 0;

            foreach (var cls in classes)
            {
                var cm = ConfusionMatrix.CalculateConfusionForOne(actual, predicted, cls);
                if ((cm.TP + cm.FN) > 0)
                    totalRecall += (double)cm.TP / (cm.TP + cm.FN);
            }

            // Macro Average: Sum of scores / Number of classes
            return totalRecall / classes.Count;
        }
    }

    public class Precision : IEvaluation
    {
        public string Name => "Precision";
        public double Calculate(double[] actual, double[] predicted)
        {
            var classes = ConfusionMatrix.GetUniqueClasses(actual);
            double totalPrecision = 0;

            foreach (var cls in classes)
            {
                var cm = ConfusionMatrix.CalculateConfusionForOne(actual, predicted, cls);
                if ((cm.TP + cm.FP) > 0)
                    totalPrecision += (double)cm.TP / (cm.TP + cm.FP);
            }

            return totalPrecision / classes.Count;
        }
    }

    public class F1Score : IEvaluation
    {
        public string Name => "F1 Score (Macro)";
        public double Calculate(double[] actual, double[] predicted)
        {
            var classes = ConfusionMatrix.GetUniqueClasses(actual);
            double totalF1 = 0;

            foreach (var cls in classes)
            {
                var cm = ConfusionMatrix.CalculateConfusionForOne(actual, predicted, cls);
                double precision = (cm.TP + cm.FP) > 0 ? (double)cm.TP / (cm.TP + cm.FP) : 0;
                double recall = (cm.TP + cm.FN) > 0 ? (double)cm.TP / (cm.TP + cm.FN) : 0;

                if (precision + recall > 0)
                    totalF1 += 2 * (precision * recall) / (precision + recall);
            }

            return totalF1 / classes.Count;
        }
    }

    public class Accuracy : IEvaluation
    {
        public string Name => "Accuracy";
        public double Calculate(double[] actual, double[] predicted)
        {
            int correct = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                if (Math.Abs(actual[i] - predicted[i]) < 0.1) correct++;
            }
            return (double)correct / actual.Length;
        }
    }

}
    

