using System;
using System.Data;
using mlDataSet;



namespace layer
{

    public interface ILossFunction
    {
        double CalculateLoss(double predicted, double actual);

        double CalculateDerivative(double predicted, double actual);
    }
    public class BinaryCrossEntropy : ILossFunction
    {
        // We need a tiny number (epsilon) to prevent Log(0) or Division by Zero
        private const double Epsilon = 1e-15;

        public double CalculateLoss(double predicted, double actual)
        {
            // Clamp value to safe range [0.000...1, 0.999...9]
            double p = Math.Clamp(predicted, Epsilon, 1.0 - Epsilon);

            // Formula: -(y * log(p) + (1-y) * log(1-p))
            return -(actual * Math.Log(p) + (1.0 - actual) * Math.Log(1.0 - p));
        }

        public double CalculateDerivative(double predicted, double actual)
        {
            double p = Math.Clamp(predicted, Epsilon, 1.0 - Epsilon);

            // Formula: (p - y) / (p * (1 - p))
            // This is derived from: -(y/p) + (1-y)/(1-p)
            return (p - actual) / (p * (1.0 - p));
        }
    }

    public class MeanSquaredError : ILossFunction
    {
        public double CalculateLoss(double predicted, double actual)
        {
            return Math.Pow(predicted - actual, 2);
        }

        public double CalculateDerivative(double predicted, double actual)
        {
            // Gradient = 2 * (y_pred - y_true)
            return 2 * (predicted - actual);
        }
    }
   
    public abstract class SimpleLayer
    {
        public double[] input { get; set; }
        public double[] output { get; set; }

        public abstract double[] Forward(double[] input);
        public abstract double[] Backward(double[] outputGradient,double learningRate);


    }

    public class DenseLayer : SimpleLayer
    {

        public double[,] Weights;
        public double[] Biases;
        private int _inputSize;
        private int _outputSize;


        public DenseLayer(int inputSize, int outputSize)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;
            Weights = new double[outputSize, inputSize];
            Biases = new double[outputSize];
            InitializeWeights();
        }

        private void InitializeWeights()
        {
            var rnd = new Random();
            for (int i = 0; i < _outputSize; i++)
            {
                Biases[i] = 0.0;
                for (int j = 0; j < _inputSize; j++)
                {
                    // Xavier/Glorot Initialization helps with vanishing gradients problem 
                    Weights[i, j] = (rnd.NextDouble() * 2 - 1) / Math.Sqrt(_inputSize);
                }
            }
        }

        public override double[] Backward(double[] outputGradient, double learningRate)
        {
            var currentLayerGradient = new double[_inputSize];

            for (int i = 0; i < _outputSize ; i++) {
                double dLoss = outputGradient[i];
                
                for (int j = 0; j < _inputSize ; j++)
                {
                    currentLayerGradient[j] += Weights[i, j] * dLoss;

                    double dLoss_Weight = dLoss * input[j];

                    Weights[i,j] -= learningRate * dLoss_Weight;
                }
                Biases[i] -= learningRate * dLoss;
                
            }
            return currentLayerGradient;



        }

        /*
         The formula for a forward pass is 
           Output = Weights * input + bias
           (m,n) * (n,1) = (m,1) + (m,1)
                 
         */
        public override double[] Forward(double[] input)
        {
            this.input = input;
            this.output = new double[_outputSize];

            for (int i = 0; i < _outputSize; i++)
            {
                double sum = Biases[i];
                
                for (int j = 0; j < _inputSize; j++)
                {
                    sum += Weights[i, j] * input[j];
                }
                this.output[i] = sum;
            }
            return this.output;
        }
    }


    public class SigmoidLayer : SimpleLayer
    {
        public override double[] Backward(double[] outputGradient, double learningRate)
        {   
            var currentLayerGradient = new double[input.Length];
            for (int i=0; i < input.Length; i++)
            {
                double s = this.output[i];
                // Derrivative of sigmoid
                double derrivative = s * (1.0 - s);
                currentLayerGradient[i] = derrivative * outputGradient[i];
            }
            return currentLayerGradient;
        }

        public override double[] Forward(double[] input)
        {
            this.input = input;
            this.output = input.Select(x=> 1.0/ (1.0 + Math.Exp(-x))).ToArray();
            return this.output;
        }
    }

    public class ReLULayer : SimpleLayer
    {
        public override double[] Forward(double[] input)
        {
            this.input = input;
            this.output = input.Select(x => Math.Max(0, x)).ToArray();
            return this.output;
        }

        public override double[] Backward(double[] nextLayerGradient, double learningRate)
        {
            var currentLayerGradient = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                // Derivative is 1 if Input > 0, else 0
                double derivative = output[i] > 0 ? 1.0 : 0.0;
                currentLayerGradient[i] = nextLayerGradient[i] * derivative;
            }
            return currentLayerGradient;
        }
    }



}
