using System;

namespace mlDataSet
{
    public class DataPoint
    {
        public double[] Features { get; set; }
        public double Label { get; set; } 
    }

    public class MLDataSet
    {
        public List<DataPoint> Points { get; set; } = new List<DataPoint>();
        public List<string> FeatureNames { get; set; } = new List<string>();

        // Method to load data from a CSV stream (compatible with Blazor InputFile)
        public static async Task<MLDataSet> LoadFromStreamAsync(Stream fileStream)
        {
            var dataset = new MLDataSet();
            using var reader = new StreamReader(fileStream);

            // Read header
            string headerLine = await reader.ReadLineAsync();
            if (string.IsNullOrEmpty(headerLine)) return dataset;

            dataset.FeatureNames = headerLine.Split(',').Select(s => s.Trim()).ToList();

            // Read data rows
            while (!reader.EndOfStream)
            {
                var line = await reader.ReadLineAsync();
                if (string.IsNullOrWhiteSpace(line)) continue;

                var parts = line.Split(',');
                // Assuming last column is Label, rest are Features
                var features = parts.Take(parts.Length - 1).Select(double.Parse).ToArray();
                var label = double.Parse(parts.Last());

                dataset.Points.Add(new DataPoint { Features = features, Label = label });
            }
            return dataset;
        }

        // Method to split data into Training and Testing sets
        public (MLDataSet Train, MLDataSet Test) Split(double trainPercentage)
        {
            var rnd = new Random();
            var shuffled = Points.OrderBy(x => rnd.Next()).ToList();
            int trainCount = (int)(shuffled.Count * trainPercentage);

            return (
                new MLDataSet { FeatureNames = this.FeatureNames, Points = shuffled.Take(trainCount).ToList() },
                new MLDataSet { FeatureNames = this.FeatureNames, Points = shuffled.Skip(trainCount).ToList() }
            );
        }
    }
}
