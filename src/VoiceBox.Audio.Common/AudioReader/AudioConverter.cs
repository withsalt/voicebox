using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace VoiceBox.Audio.Common.AudioReader
{
    public static class AudioConverter
    {
        /// <summary>
        /// 将 float 样本数组转换为 16-bit 整型 (PCM) 字节数组
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public static byte[] ConvertTo16BitInt(float[] data)
        {
            if (data == null || data.Length == 0)
                return new byte[0];

            byte[] byteBuffer = new byte[data.Length * 2];
            int bufferIndex = 0;

            for (int i = 0; i < data.Length; i++)
            {
                float sample = data[i];
                if (sample > 1.0f) sample = 1.0f;
                if (sample < -1.0f) sample = -1.0f;

                short intSample = (short)(sample * short.MaxValue);

                byteBuffer[bufferIndex++] = (byte)(intSample & 0xFF);
                byteBuffer[bufferIndex++] = (byte)(intSample >> 8);
            }

            return byteBuffer;
        }

        /// <summary>
        /// 将 float 样本数组转换为 32-bit 浮点字节数组。
        /// </summary>
        /// <param name="data">源 float 数组。</param>
        /// <returns>代表 32-bit 浮点音频的 byte 数组。</returns>
        public static byte[] ConvertTo32BitFloat(float[] data)
        {
            if (data == null || data.Length == 0)
                return new byte[0];

            byte[] byteBuffer = new byte[data.Length * 4];
            Buffer.BlockCopy(data, 0, byteBuffer, 0, byteBuffer.Length);
            return byteBuffer;
        }

        /// <summary>
        /// 将 float 样本数组转换为 32-bit 整型 (PCM) 字节数组（仅在特定场景使用）。
        /// </summary>
        /// <param name="data">源 float 数组。</param>
        /// <returns>代表 32-bit 整型 PCM 音频的 byte 数组。</returns>
        public static byte[] ConvertTo32BitInt(float[] data)
        {
            if (data == null || data.Length == 0)
                return new byte[0];

            byte[] byteBuffer = new byte[data.Length * 4];
            int bufferIndex = 0;

            const double maxVal = int.MaxValue;

            for (int i = 0; i < data.Length; i++)
            {
                float sample = data[i];
                if (sample > 1.0f) sample = 1.0f;
                if (sample < -1.0f) sample = -1.0f;

                int intSample = (int)(sample * maxVal);

                // 3. 使用位操作直接写入 4 个字节 (Little-Endian)
                byteBuffer[bufferIndex++] = (byte)(intSample & 0xFF);         // Byte 0
                byteBuffer[bufferIndex++] = (byte)(intSample >> 8 & 0xFF);  // Byte 1
                byteBuffer[bufferIndex++] = (byte)(intSample >> 16 & 0xFF); // Byte 2
                byteBuffer[bufferIndex++] = (byte)(intSample >> 24);          // Byte 3
            }

            return byteBuffer;
        }

        /// <summary>
        /// 将PCM音频数据转换为32位浮点数组
        /// </summary>
        /// <param name="data">原始PCM字节数据</param>
        /// <param name="bitsPerSample">每个采样的位数 (16, 24, 32)</param>
        /// <param name="isLittleEndian">是否小端字节序（默认true）</param>
        /// <param name="isFloat">是否为浮点格式（默认false，整数格式）</param>
        /// <param name="normalize">是否归一化到[-1, 1]范围（默认true）</param>
        /// <returns>转换后的浮点数组</returns>
        public static float[] ConvertToFloat32(byte[] data, int bitsPerSample, bool isLittleEndian = true, bool isFloat = false, bool normalize = true)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentException("PCM数据不能为空");

            int bytesPerSample = bitsPerSample / 8;
            if (data.Length % bytesPerSample != 0)
                throw new ArgumentException("数据长度与样本大小不匹配");

            int sampleCount = data.Length / bytesPerSample;
            float[] floatSamples = new float[sampleCount];

            if (isFloat && bitsPerSample == 32)
            {
                ConvertFloat32(data, floatSamples, isLittleEndian);
                return floatSamples;
            }

            // The original implementation calculated this invariant for every sample.
            // Keeping the calculation in double preserves the existing numeric result.
            double maxValue = normalize ? Math.Pow(2, bitsPerSample - 1) : 1d;
            if (bitsPerSample is 8 or 16 or 24 or 32)
            {
                ConvertIntegerSamples(data, floatSamples, bytesPerSample, isLittleEndian, normalize, maxValue);
                return floatSamples;
            }

            for (int i = 0; i < sampleCount; i++)
            {
                int offset = i * bytesPerSample;
                long intValue = ConvertBytesToInteger(data, offset, bitsPerSample, isLittleEndian);
                floatSamples[i] = normalize ? (float)(intValue / maxValue) : intValue;
            }

            return floatSamples;
        }

        private static void ConvertIntegerSamples(
            ReadOnlySpan<byte> source,
            Span<float> destination,
            int bytesPerSample,
            bool isLittleEndian,
            bool normalize,
            double maxValue)
        {
            switch (bytesPerSample)
            {
                case 1:
                    for (int i = 0; i < destination.Length; i++)
                    {
                        destination[i] = normalize ? (float)(source[i] / maxValue) : source[i];
                    }
                    return;
                case 2:
                    ConvertInt16Samples(source, destination, isLittleEndian, normalize, maxValue);
                    return;
                case 3:
                    for (int i = 0, offset = 0; i < destination.Length; i++, offset += 3)
                    {
                        int value = isLittleEndian
                            ? source[offset] | source[offset + 1] << 8 | source[offset + 2] << 16
                            : source[offset] << 16 | source[offset + 1] << 8 | source[offset + 2];
                        if ((value & 0x800000) != 0)
                        {
                            value |= unchecked((int)0xFF000000);
                        }

                        destination[i] = normalize ? (float)(value / maxValue) : value;
                    }
                    return;
                case 4:
                    ConvertInt32Samples(source, destination, isLittleEndian, normalize, maxValue);
                    return;
            }
        }

        private static void ConvertInt16Samples(
            ReadOnlySpan<byte> source,
            Span<float> destination,
            bool isLittleEndian,
            bool normalize,
            double maxValue)
        {
            if (isLittleEndian == BitConverter.IsLittleEndian)
            {
                ReadOnlySpan<short> samples = MemoryMarshal.Cast<byte, short>(source);
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] = normalize ? (float)(samples[i] / maxValue) : samples[i];
                }
                return;
            }

            for (int i = 0, offset = 0; i < destination.Length; i++, offset += sizeof(short))
            {
                short value = isLittleEndian
                    ? BinaryPrimitives.ReadInt16LittleEndian(source[offset..])
                    : BinaryPrimitives.ReadInt16BigEndian(source[offset..]);
                destination[i] = normalize ? (float)(value / maxValue) : value;
            }
        }

        private static void ConvertInt32Samples(
            ReadOnlySpan<byte> source,
            Span<float> destination,
            bool isLittleEndian,
            bool normalize,
            double maxValue)
        {
            if (isLittleEndian == BitConverter.IsLittleEndian)
            {
                ReadOnlySpan<int> samples = MemoryMarshal.Cast<byte, int>(source);
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] = normalize ? (float)(samples[i] / maxValue) : samples[i];
                }
                return;
            }

            for (int i = 0, offset = 0; i < destination.Length; i++, offset += sizeof(int))
            {
                int value = isLittleEndian
                    ? BinaryPrimitives.ReadInt32LittleEndian(source[offset..])
                    : BinaryPrimitives.ReadInt32BigEndian(source[offset..]);
                destination[i] = normalize ? (float)(value / maxValue) : value;
            }
        }

        /// <summary>
        /// 处理整数格式的转换
        /// </summary>
        private static long ConvertBytesToInteger(byte[] data, int offset, int bits, bool isLittleEndian)
        {
            if (offset + bits / 8 > data.Length)
                throw new IndexOutOfRangeException();

            int byteCount = bits / 8;
            long result = 0;

            if (isLittleEndian)
            {
                // 小端字节序处理
                for (int i = 0; i < byteCount; i++)
                {
                    result |= (long)data[offset + i] << 8 * i;
                }
            }
            else
            {
                // 大端字节序处理
                for (int i = 0; i < byteCount; i++)
                {
                    result = result << 8 | data[offset + i];
                }
            }

            // 符号扩展处理（对于有符号整数）
            if (bits > 8 && (data[isLittleEndian ? offset + byteCount - 1 : offset] & 0x80) != 0)
            {
                result |= ~0L << bits - 1;
            }

            return result;
        }

        private static void ConvertFloat32(byte[] data, Span<float> destination, bool isLittleEndian)
        {
            if (isLittleEndian == BitConverter.IsLittleEndian)
            {
                MemoryMarshal.Cast<byte, float>(data).CopyTo(destination);
                return;
            }

            ReadOnlySpan<byte> source = data;
            for (int i = 0, offset = 0; i < destination.Length; i++, offset += sizeof(float))
            {
                destination[i] = isLittleEndian
                    ? BinaryPrimitives.ReadSingleLittleEndian(source[offset..])
                    : BinaryPrimitives.ReadSingleBigEndian(source[offset..]);
            }
        }



    }
}
