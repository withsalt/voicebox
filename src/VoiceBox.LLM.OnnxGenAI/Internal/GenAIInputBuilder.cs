using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntimeGenAI;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.OnnxGenAI.Internal
{
    /// <summary>
    /// 把 <see cref="ChatMessage"/> 中的多模态内容拆解为 ORT-GenAI 期望的输入对象:
    /// <list type="bullet">
    ///   <item><c>prompt</c>: 由 <see cref="GenAIPromptTemplate"/> 渲染后的字符串。</item>
    ///   <item><c>audios</c>: <see cref="Audios"/> 实例 (零或一个 — Gemma 4 限制), <c>null</c> 表示无音频。</item>
    ///   <item><c>images</c>: <see cref="Images"/> 实例, <c>null</c> 表示无图像。</item>
    /// </list>
    ///
    /// 对调用方暴露的 <see cref="BuildResult"/> 是 <see cref="IDisposable"/>, 必须在推理完成后释放,
    /// 因为 <see cref="Audios"/> / <see cref="Images"/> 持有 native 句柄。
    /// </summary>
    internal static class GenAIInputBuilder
    {
        /// <summary>
        /// Gemma 4 当前实现只允许每个 prompt 一个音频片段
        /// (见 onnxruntime-genai/src/models/gemma4_multimodal_processor.cpp 的 audio_marker_count 校验)。
        /// </summary>
        private const int MaxAudioClipsPerPrompt = 1;

        public static BuildResult Build(
            GenAIPromptTemplate template,
            Tokenizer tokenizer,
            IReadOnlyList<ChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            var (audioPaths, audioBytes, imagePaths, imageBytes) = CollectMedia(messages);

            int totalAudio = (audioPaths?.Count ?? 0) + (audioBytes?.Count ?? 0);
            int totalImage = (imagePaths?.Count ?? 0) + (imageBytes?.Count ?? 0);

            if (totalAudio > MaxAudioClipsPerPrompt)
            {
                throw new System.NotSupportedException(
                    $"Gemma 4 currently supports at most {MaxAudioClipsPerPrompt} audio clip per prompt, " +
                    $"got {totalAudio}.");
            }
            ValidateImageSourceCompatibility(imagePaths?.Count ?? 0, imageBytes?.Count ?? 0);

            var rendered = template.Render(tokenizer, messages, addGenerationPrompt, enableThinking);

            if (rendered.AudioCount != totalAudio)
            {
                throw new InvalidOperationException(
                    $"Internal error: prompt has {rendered.AudioCount} audio placeholders but {totalAudio} audio clips were collected.");
            }
            if (rendered.ImageCount != totalImage)
            {
                throw new InvalidOperationException(
                    $"Internal error: prompt has {rendered.ImageCount} image placeholders but {totalImage} images were collected.");
            }

            Audios? audios = null;
            Images? images = null;
            try
            {
                audios = LoadAudios(audioPaths, audioBytes);
                images = LoadImages(imagePaths, imageBytes);
                return new BuildResult(rendered.Prompt, audios, images);
            }
            catch
            {
                audios?.Dispose();
                images?.Dispose();
                throw;
            }
        }

        private static (List<string>? audioPaths, List<byte[]>? audioBytes, List<string>? imagePaths, List<byte[]>? imageBytes)
            CollectMedia(IReadOnlyList<ChatMessage> messages)
        {
            List<string>? audioPaths = null;
            List<byte[]>? audioBytes = null;
            List<string>? imagePaths = null;
            List<byte[]>? imageBytes = null;

            foreach (var msg in messages)
            {
                foreach (var item in msg.Content)
                {
                    switch (item)
                    {
                        case AudioContent a when a.Bytes is not null:
                            (audioBytes ??= new List<byte[]>(1)).Add(a.Bytes);
                            break;
                        case AudioContent a when !string.IsNullOrEmpty(a.Path):
                            (audioPaths ??= new List<string>(1)).Add(a.Path!);
                            break;
                        case AudioContent:
                            throw new ArgumentException("AudioContent must have either Path or Bytes set.");

                        case ImageContent i when i.Bytes is not null:
                            (imageBytes ??= new List<byte[]>(1)).Add(i.Bytes);
                            break;
                        case ImageContent i when !string.IsNullOrEmpty(i.Path):
                            (imagePaths ??= new List<string>(1)).Add(i.Path!);
                            break;
                        case ImageContent:
                            throw new ArgumentException("ImageContent must have either Path or Bytes set.");
                    }
                }
            }

            return (audioPaths, audioBytes, imagePaths, imageBytes);
        }

        private static Audios? LoadAudios(List<string>? paths, List<byte[]>? bytes)
        {
            if (paths is { Count: > 0 })
                return Audios.Load(paths.ToArray());
            if (bytes is { Count: > 0 })
                return Audios.Load(bytes[0]);  // 上层已校验最多 1 个
            return null;
        }

        private static Images? LoadImages(List<string>? paths, List<byte[]>? bytes)
        {
            ValidateImageSourceCompatibility(paths?.Count ?? 0, bytes?.Count ?? 0);

            if (paths is { Count: > 0 })
                return Images.Load(paths.ToArray());
            if (bytes is { Count: > 0 })
                return Images.Load(bytes[0]);  // ORT-GenAI byte[] API 当前每次仅支持 1 张
            return null;
        }

        internal static void ValidateImageSourceCompatibility(int pathCount, int byteCount)
        {
            if (pathCount > 0 && byteCount > 0)
            {
                throw new NotSupportedException(
                    "ImageContent cannot mix file paths and byte arrays in one prompt because " +
                    "ORT-GenAI exposes Images.Load(string[]) for multiple files and Images.Load(byte[]) for a single in-memory image.");
            }

            if (byteCount > 1)
            {
                throw new NotSupportedException(
                    "ORT-GenAI Images.Load(byte[]) supports only one in-memory image per prompt. " +
                    "Use image file paths for multiple images.");
            }
        }

        /// <summary>
        /// InputBuilder 的输出。<see cref="Audios"/>/<see cref="Images"/> 的所有权移交调用方,
        /// 必须 <see cref="Dispose"/> 释放 native 句柄。
        /// </summary>
        internal sealed class BuildResult : IDisposable
        {
            public BuildResult(string prompt, Audios? audios, Images? images)
            {
                Prompt = prompt;
                Audios = audios;
                Images = images;
            }

            public string Prompt { get; }

            public Audios? Audios { get; }

            public Images? Images { get; }

            public void Dispose()
            {
                Audios?.Dispose();
                Images?.Dispose();
            }
        }
    }
}
