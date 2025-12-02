using System;
using System.Collections;
using layer;
using mlDataSet;


namespace Models
{
    public interface IClassifier
    {
        string Name { get; }
        Dictionary<string, double> Hyperparameters { get; }
        void Train(MLDataSet trainData);
        double Predict(double[] features);
    }

    public class KNNClassifier : IClassifier
    {
        public string Name => "K-Nearest Neighbors";

        public Dictionary<string, double> Hyperparameters { get; } = new Dictionary<string, double>
        {
            { "K", 3 }
        };

        private List<DataPoint> _trainingData;

        public void Train(MLDataSet trainData)
        {
            _trainingData = trainData.Points;
        }

        public double Predict(double[] features)
        {
            if (_trainingData == null || _trainingData.Count == 0) return 0.0;

            int k = (int)Hyperparameters["K"];

            // 1. Calculate distance from input to EVERY training point
            // 2. Sort by distance (smallest first)
            // 3. Take the top K
            var nearestNeighbors = _trainingData
                .Select(point => new
                {
                    Point = point,
                    Distance = EuclideanDistance(features, point.Features)
                })
                .OrderBy(x => x.Distance)
                .Take(k)
                .ToList();

            if (!nearestNeighbors.Any()) return 0.0;

            // 4. Majority Vote: Group by Label and count them
            var vote = nearestNeighbors
                .GroupBy(x => x.Point.Label)
                .OrderByDescending(g => g.Count())
                .First()
                .Key;

            return vote;
        }

        private double EuclideanDistance(double[] a, double[] b)
        {
            double sum = 0;
            // Assuming a and b are same length
            for (int i = 0; i < a.Length; i++)
            {
                sum += Math.Pow(a[i] - b[i], 2);
            }
            return Math.Sqrt(sum);
        }
    }
    public class NaiveBayes : IClassifier
    {
        private double[] mean0, mean1, var0, var1;
        private double prior0, prior1;
        public string Name => "Naive-Bayes";

        public Dictionary<string, double> Hyperparameters => new Dictionary<string, double> {};

        /// <summary>
        /// Words = [x1,x2,x2,...,xn] 
        /// We know that P(Feature1| Words) = P(Words| P(Features) / P(Words)
        /// We assume words are independent so the equation above becomes
        /// P(x1,Words) * P(x2,Words
        /// </summary>
        /// <param name="features"></param>
        /// <returns></returns>
        public double Predict(double[] features)
        {
            double p0 = Math.Log(prior0); double p1 = Math.Log(prior1);


            
            for (int i = 0; i < features.Length; i++)
            {
                p0 += Math.Log(gauss(features[i], mean0[i], var0[i]));
                p1 += Math.Log(gauss(features[i], mean1[i], var1[i]));
            }
            return p1 > p0 ? 1.0 : 0.0;
        }
        private double gauss(double x, double m, double v)
        {
            // We add a small number to ensure varriance different from 0
            return (1 / Math.Sqrt(2 * Math.PI * (v + 1e-9))) * Math.Exp(-(Math.Pow(x - m, 2) / (2 * (v + 1e-9))));
        }

        public void Train(MLDataSet trainData)
        {
            var points = trainData.Points;
            int nr_of_features = points[0].Features.Length;
            int totalSamples = points.Count;
            List<double[]> c0 = new List<double[]>();
            List<double[]> c1 = new List<double[]>();

            points.ForEach(p =>
            {
                if (p.Label == 0)
                    c0.Add(p.Features);
                else
                    c1.Add(p.Features);
            });

            prior0 = (double) c0.Count / totalSamples; 
            prior1 = (double) c1.Count / totalSamples;

            mean0 = computeMean(c0, nr_of_features); var0 = computeVariance(c0, mean0, nr_of_features);
            mean1 = computeMean(c1, nr_of_features); var1 = computeVariance(c1 , mean1, nr_of_features);


        }

        private double[] computeMean(List<double[]> list,int n) {

            double[] mean = new double[n];

            foreach (var d in list)
                for (int i = 0; i < n; i++)
                    mean[i] += d[i];


            for (int i = 0; i < n; i++)
                mean[i] /= Math.Max(1, list.Count);

            return mean;
        
        }

        private double[] computeVariance(List<double[]> list,double[]mean, int n)
        {
            double[] variance = new double[n];

            list.ForEach(d =>
            {
                for (int i = 0; i < n; i++)
                    variance[i] += Math.Pow(d[i] - mean[i], 2);

            });
            for (int i=0; i < n; i++)
            {
                variance[i] /= Math.Max(0, list.Count);
            }
            return variance;
        }


    }



    public class SequentialModel : IClassifier
    {
        public ILossFunction LossFunction { get; set; }
        public string Name { get; set; } = "Deep Neural Network";
        public List<SimpleLayer> Layers { get; set; } = new List<SimpleLayer>();

        public Dictionary<string, double> Hyperparameters { get; } = new Dictionary<string, double>  {
            { "LearningRate", 0.01 },
            { "Epochs", 100 }
        };

        public SequentialModel(ILossFunction lossFunction = null)
        {
            LossFunction = lossFunction ?? new MeanSquaredError();
        }

        public void Add(SimpleLayer layer) => Layers.Add(layer);

        public double[] PredictRaw(double[] input)
        {
            double[] signal = input;
            foreach (var layer in Layers)
            {
                signal = layer.Forward(signal);
            }
            return signal;
        }

        public double Predict(double[] features)
        {
            var output = PredictRaw(features);
            // If single output, treat as binary class
            if (output.Length == 1) return output[0] >= 0.5 ? 1.0 : 0.0;

            // If multiple outputs, return index of max
            double maxVal = double.NegativeInfinity;
            int maxIndex = 0;
            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] > maxVal) { maxVal = output[i]; maxIndex = i; }
            }
            return (double)maxIndex;
        }

        public void Train(MLDataSet trainData)
        {
            int epochs = (int)Hyperparameters["Epochs"];
            double lr = Hyperparameters["LearningRate"];

            for (int e = 0; e < epochs; e++)
            {
                double totalError = 0;

                foreach (var point in trainData.Points)
                {
                    // 1. Forward Pass
                    var prediction = PredictRaw(point.Features);

                    // 2. Compute Initial Gradients
                    double[] outputGradient = new double[prediction.Length];

                    for (int i = 0; i < prediction.Length; i++)
                    {
                        // Determine Target
                        double target;
                        if (prediction.Length == 1)
                        {
                            target = point.Label;
                        }
                        else
                        {
                            // Multi-class: One-hot encoding
                            target = (i == (int)point.Label) ? 1.0 : 0.0;
                        }

                        outputGradient[i] = LossFunction.CalculateDerivative(prediction[i], target);

                        totalError += LossFunction.CalculateLoss(prediction[i], target);
                    }

                    // 3. Backward Pass
                    double[] signal = outputGradient;
                    for (int i = Layers.Count - 1; i >= 0; i--)
                    {
                        signal = Layers[i].Backward(signal, lr);
                    }
                }
            }
        }


    }

}