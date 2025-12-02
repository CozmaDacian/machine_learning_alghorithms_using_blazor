using mlDataSet;
using System;

namespace DataInterpretation
{

    public class DataFrame
    {

        private Dictionary<string,List<object>> _collumns;
        public List<string> CollumnNames { get; private set; }

        public int RowCount => _collumns.Count > 0 ? _collumns.First().Value.Count : 0;

        public DataFrame()
        {
            _collumns = new Dictionary<string, List<object>>();
            CollumnNames = new List<string>();
        }
        public List<object> this[string collumnName]
        {
            get
            {
                if (_collumns.ContainsKey(collumnName))
                    return _collumns[collumnName];
                throw new ArgumentException($"Column '{collumnName}' not found");
            }
        }

        public static async Task<DataFrame> ReadCsvAsync(string filePath)
        {
            using var stream = File.OpenRead(filePath);
            return await ReadCsvStreamAsync(stream);
        }

        public static async Task<DataFrame> ReadCsvStreamAsync(Stream fileStream)
        {
            var df = new DataFrame();
            using var reader = new StreamReader(fileStream);

            string headerLine = await reader.ReadLineAsync();
            if (string.IsNullOrEmpty(headerLine)) return df;

            df.CollumnNames = headerLine.Split(',').Select(h => h.Trim()).ToList();

            foreach (var col in df.CollumnNames)
            {
                df._collumns[col] = new List<object>();
            }
            string line;
            while ((line = await reader.ReadLineAsync()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                var values = line.Split(',');

                if (values.Length != df.CollumnNames.Count) continue; // Skip malformed rows

                for (int i = 0; i < values.Length; i++)
                {
                    string rawVal = values[i].Trim();
                    string colName = df.CollumnNames[i];

                    if (double.TryParse(rawVal, out double dVal))
                    {
                        df._collumns[colName].Add(dVal);
                    }
                    else
                    {
                        df._collumns[colName].Add(rawVal); // Keep as string
                    }
                }
            }

            return df;
        }


        public void Head(int count = 5)
        {
            int limit = Math.Min(count, RowCount);
            Console.WriteLine(string.Join(" | ", CollumnNames));
            Console.WriteLine(new string('-', 50));

            for (int i = 0; i < limit; i++)
            {
                var row = CollumnNames.Select(c => _collumns[c][i].ToString());
                Console.WriteLine(string.Join(" | ", row));
            }
        }

        // Helper to convert this generic Frame into your specific MLDataSet for training
        public MLDataSet ToMLDataSet(string labelColumnName, params string[] featureColumnNames)
        {
            var dataset = new MLDataSet();
            dataset.FeatureNames = featureColumnNames.ToList();

            for (int i = 0; i < RowCount; i++)
            {
                var features = new double[featureColumnNames.Length];

                for (int j = 0; j < featureColumnNames.Length; j++)
                {
                    var val = _collumns[featureColumnNames[j]][i];
                    features[j] = Convert.ToDouble(val);
                }

                var labelVal = _collumns[labelColumnName][i];
                double label = Convert.ToDouble(labelVal);

                dataset.Points.Add(new DataPoint
                {
                    Features = features,
                    Label = label
                });
            }

            return dataset;
        }
    }
}
    


