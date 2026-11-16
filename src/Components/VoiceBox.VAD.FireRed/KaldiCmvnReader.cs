namespace VoiceBox.VAD.FireRed
{
    /// <summary>
    /// 读取 Kaldi 二进制 .ark 中的 CMVN 统计矩阵（DM 或 FM 格式），并计算 means / 1/std。
    /// 矩阵形状为 (2, dim+1)：行0为 sum + count；行1为 sum_sq + 0。
    /// </summary>
    internal static class KaldiCmvnReader
    {
        public static (float[] Means, float[] InvStd) Load(string arkPath)
        {
            using var fs = File.OpenRead(arkPath);
            using var br = new BinaryReader(fs);

            byte b0 = br.ReadByte();
            // 若文件包含可读 key（非 \0 起始），跳过到第一个空格
            if (b0 != 0)
            {
                while (br.ReadByte() != ' ') { }
                b0 = br.ReadByte();
            }
            if (b0 != 0)
                throw new InvalidDataException("Expected Kaldi binary marker (\\0B).");
            if ((char)br.ReadByte() != 'B')
                throw new InvalidDataException("Expected 'B' after binary marker.");

            char m0 = (char)br.ReadByte();
            char m1 = (char)br.ReadByte();
            char m2 = (char)br.ReadByte();
            bool isDouble;
            if (m0 == 'D' && m1 == 'M' && m2 == ' ') isDouble = true;
            else if (m0 == 'F' && m1 == 'M' && m2 == ' ') isDouble = false;
            else throw new InvalidDataException($"Unsupported Kaldi matrix type: '{m0}{m1}{m2}'.");

            if (br.ReadByte() != 4)
                throw new InvalidDataException("Expected size byte 4 before rows.");
            int rows = br.ReadInt32();
            if (br.ReadByte() != 4)
                throw new InvalidDataException("Expected size byte 4 before cols.");
            int cols = br.ReadInt32();

            if (rows != 2)
                throw new InvalidDataException($"Expected 2 rows in CMVN stats, got {rows}.");
            if (cols < 2)
                throw new InvalidDataException($"Expected at least 2 cols in CMVN stats, got {cols}.");

            int dim = cols - 1;
            var stats = new double[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    stats[r, c] = isDouble ? br.ReadDouble() : br.ReadSingle();
                }
            }

            double count = stats[0, dim];
            if (count <= 0)
                throw new InvalidDataException("Invalid CMVN count.");

            var means = new float[dim];
            var invStd = new float[dim];
            for (int i = 0; i < dim; i++)
            {
                double mean = stats[0, i] / count;
                double var = stats[1, i] / count - mean * mean;
                if (var < 1e-20) var = 1e-20;
                means[i] = (float)mean;
                invStd[i] = (float)(1.0 / Math.Sqrt(var));
            }

            return (means, invStd);
        }
    }
}
