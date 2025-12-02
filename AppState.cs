using mlDataSet;
using Models;

namespace machine_learnintg.Services
{
    public class AppState
    {
        // The processed dataset ready for training
        public MLDataSet CurrentDataSet { get; private set; }

        // The raw dataframe (optional, if you want to keep it)
        public DataInterpretation.DataFrame RawDataFrame { get; private set; }

        public event Action OnChange;

        public void SetDataSet(MLDataSet dataSet)
        {
            CurrentDataSet = dataSet;
            NotifyStateChanged();
        }

        public void SetDataFrame(DataInterpretation.DataFrame df)
        {
            RawDataFrame = df;
            NotifyStateChanged();
        }

        private void NotifyStateChanged() => OnChange?.Invoke();
    }
}