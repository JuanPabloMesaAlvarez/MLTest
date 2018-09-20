using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLTest.Consoles
{
    public class Abalone
    {
        [Column("0")]
        public string Sex;

        [Column("1")]
        public float Lenght;

        [Column("2")]
        public float Diameter;

        [Column("3")]
        public float Height;

        [Column("4")]
        public float WholeWeight;

        [Column("5")]
        public float ShuckedWeight;

        [Column("6")]
        public float VisceraWeight;

        [Column("7")]
        public float ShellWeight;

        [Column("8")]
        public float Age;
    }

    public class AbaloneAgePrediction
    {
        [ColumnName("Score")]
        public float Age;
    }
}
