using mlDataSet;
using DataInterpretation;
using System;
using System.Collections.Generic;

namespace machine_learnintg.Services
{
    // 1. Define this class here so it is accessible by both the Service and the Pages
    public class ColumnConfig
    {
        public string Name { get; set; }
        public object PreviewValue { get; set; }
        public bool IsFeature { get; set; } = true;
        public bool IsLabel { get; set; } = false;
    }

    public class AppState
    {
        // 2. The Final Dataset (Ready for training)
        public MLDataSet CurrentDataSet { get; private set; }

        // 3. The Raw Data (Persists the uploaded CSV)
        public DataFrame RawDataFrame { get; private set; }

        // 4. The Configuration State (Persists your checkboxes/radio buttons)
        public List<ColumnConfig> ImportConfigurations { get; private set; }

        public event Action OnChange;

        public void SetDataSet(MLDataSet dataSet)
        {
            CurrentDataSet = dataSet;
            NotifyStateChanged();
        }

        // 5. New Method: Save the raw import state
        public void SaveImportState(DataFrame rawFrame, List<ColumnConfig> configs)
        {
            RawDataFrame = rawFrame;
            ImportConfigurations = configs;
            NotifyStateChanged();
        }

        private void NotifyStateChanged() => OnChange?.Invoke();
    }
}