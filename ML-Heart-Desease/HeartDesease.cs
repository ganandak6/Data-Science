using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Heart_Desease
{
    public class HeartDesease
    {
        [LoadColumn(0)]
        public float age { get; set; }
        [LoadColumn(1)]
        public float sex { get; set; }
        [LoadColumn(2)]
        public float cp { get; set; }
        [LoadColumn(3)]
        public float trestnps { get; set; }
        [LoadColumn(4)]
        public float chol { get; set; }
        [LoadColumn(5)]
        public float fbs { get; set; }
        [LoadColumn(6)]
        public float restecg { get; set; }
        [LoadColumn(7)]
        public float thalach { get; set; }
        [LoadColumn(8)]
        public float exang { get; set; }
        [LoadColumn(9)]
        public float oldpeak { get; set; }
        [LoadColumn(10)]
        public float slope { get; set; }
        [LoadColumn(11)]
        public float ca { get; set; }
        [LoadColumn(12)]
        public float thal { get; set; }
        [LoadColumn(13)]
        [ColumnName("Label")]
        //[Column(Ordinal: "13", name: "Label")]
        public bool target { get; set; }
        // public float Features { get; set; }
    }

    /*
     private class RegressionData
{
    [LoadColumn(0, 10), ColumnName("Features")]
    public float FeatureVector { get; set;}

    [LoadColumn(11)]
    public float Target { get; set;}
}*/

    public class HeartDeseaseOutput : HeartDesease
    {
        // ColumnName attribute is used to change the column name from
        // its default value, which is the name of the field.
        [ColumnName("PredictedLabel")]
        public String Prediction { get; set; }
        //public float[] Score { get; set; }
        [ColumnName("Probability")]
        public float Probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }   
}
