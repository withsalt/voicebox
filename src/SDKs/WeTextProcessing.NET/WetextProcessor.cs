using System;
using System.IO;
using System.Runtime.InteropServices;
using WeTextProcessing.NET.Interop;

namespace WeTextProcessing.NET;

/// <summary>
/// Specifies the normalization direction for <see cref="WetextProcessor"/>.
/// </summary>
public enum WetextMode
{
    /// <summary>
    /// Text Normalization: converts written form to spoken form
    /// (e.g. "2023年" → "二零二三年", "$1.5" → "one dollar and fifty cents").
    /// </summary>
    TextNormalization,

    /// <summary>
    /// Inverse Text Normalization: converts spoken form to written form
    /// (e.g. "二零二三年" → "2023年", "one dollar fifty" → "$1.50").
    /// </summary>
    InverseTextNormalization,
}

/// <summary>
/// Specifies the language for <see cref="WetextProcessor"/>.
/// </summary>
public enum WetextLanguage
{
    /// <summary>
    /// Chinese. Uses <c>zh_*.fst</c> model files.
    /// </summary>
    Chinese,

    /// <summary>
    /// English. Uses <c>en_*.fst</c> model files.
    /// </summary>
    English,
}

/// <summary>
/// Provides text normalization and processing capabilities through the WeTextProcessing native library.
/// Behavior is 1:1 with the Python <c>WeTextProcessing</c> package (default build options),
/// since both apply the same precompiled FST model files.
/// </summary>
public sealed class WetextProcessor : IDisposable
{
    /// <summary>
    /// The directory (relative to <see cref="AppContext.BaseDirectory"/>) where the default
    /// FST model files are expected to be located.
    /// </summary>
    public const string DefaultModelRelativeDirectory = "runtimes/models/WeTextProcessing";

    private readonly WetextProcessorSafeHandle _handle;
    private volatile bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="WetextProcessor"/> class for Chinese,
    /// using FST model files from the default model directory next to the running assembly
    /// (<see cref="DefaultModelRelativeDirectory"/>).
    /// </summary>
    /// <param name="mode">The normalization direction. Defaults to <see cref="WetextMode.TextNormalization"/>.</param>
    /// <exception cref="FileNotFoundException">A required FST model file could not be found in the default directory.</exception>
    /// <exception cref="WetextProcessorException">The native processor failed to initialize.</exception>
    public WetextProcessor(WetextMode mode = WetextMode.TextNormalization)
        : this(WetextLanguage.Chinese, mode)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="WetextProcessor"/> class for the given
    /// language and direction, using FST model files from the default model directory next
    /// to the running assembly (<see cref="DefaultModelRelativeDirectory"/>).
    /// </summary>
    /// <param name="language">The language whose FST models should be loaded.</param>
    /// <param name="mode">The normalization direction. Defaults to <see cref="WetextMode.TextNormalization"/>.</param>
    /// <exception cref="FileNotFoundException">A required FST model file could not be found in the default directory.</exception>
    /// <exception cref="WetextProcessorException">The native processor failed to initialize.</exception>
    public WetextProcessor(WetextLanguage language, WetextMode mode = WetextMode.TextNormalization)
        : this(GetDefaultModelPath(language, mode, "tagger"), GetDefaultModelPath(language, mode, "verbalizer"))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="WetextProcessor"/> class from explicit model paths.
    /// </summary>
    /// <param name="taggerPath">
    /// The file path to the tagger FST model. The file name must keep its original
    /// <c>zh_tn_</c>/<c>zh_itn_</c>/<c>en_tn_</c>/<c>en_itn_</c> prefix — the native library
    /// derives the token-reordering rules from it.
    /// </param>
    /// <param name="verbalizerPath">The file path to the verbalizer FST model.</param>
    /// <exception cref="ArgumentException">
    /// <paramref name="taggerPath"/> or <paramref name="verbalizerPath"/> is null or whitespace.
    /// </exception>
    /// <exception cref="FileNotFoundException">
    /// The specified FST model file does not exist.
    /// </exception>
    /// <exception cref="WetextProcessorException">
    /// The native processor failed to initialize (corrupt FST file or unrecognized file name prefix).
    /// </exception>
    public WetextProcessor(string taggerPath, string verbalizerPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(taggerPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(verbalizerPath);

        if (!File.Exists(taggerPath))
        {
            throw new FileNotFoundException(
                $"Tagger FST file not found: {taggerPath}", taggerPath);
        }

        if (!File.Exists(verbalizerPath))
        {
            throw new FileNotFoundException(
                $"Verbalizer FST file not found: {verbalizerPath}", verbalizerPath);
        }

        NativeLibraryLoader.EnsureRegistered();

        var rawHandle = NativeMethods.wetext_create_processor(taggerPath, verbalizerPath);
        if (rawHandle == IntPtr.Zero)
        {
            throw new WetextProcessorException(
                $"Failed to create WeTextProcessing processor. " +
                $"Tagger: {taggerPath}, Verbalizer: {verbalizerPath}");
        }

        _handle = new WetextProcessorSafeHandle(rawHandle);
    }

    /// <summary>
    /// Applies the tagger FST to the input text, producing tagged output.
    /// </summary>
    /// <param name="input">The input text to tag. An empty string yields an empty string.</param>
    /// <returns>The tagged text.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="input"/> is null.</exception>
    /// <exception cref="ObjectDisposedException">This instance has been disposed.</exception>
    /// <exception cref="WetextProcessorException">The native processor failed to process the input.</exception>
    public string Tag(string input)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(input);

        if (input.Length == 0) return string.Empty;
        return InvokeNative(handle => NativeMethods.wetext_tag(handle, input));
    }

    /// <summary>
    /// Applies the verbalizer FST to previously tagged text, producing the final output.
    /// </summary>
    /// <param name="input">The tagged text to verbalize. An empty string yields an empty string.</param>
    /// <returns>The verbalized (normalized) text.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="input"/> is null.</exception>
    /// <exception cref="ObjectDisposedException">This instance has been disposed.</exception>
    /// <exception cref="WetextProcessorException">
    /// <paramref name="input"/> is not well-formed tagged text, or the native processor failed.
    /// </exception>
    public string Verbalize(string input)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(input);

        if (input.Length == 0) return string.Empty;
        return InvokeNative(handle => NativeMethods.wetext_verbalize(handle, input));
    }

    /// <summary>
    /// Performs full text normalization: tag then verbalize in a single call.
    /// </summary>
    /// <param name="input">The input text to normalize. An empty string yields an empty string.</param>
    /// <returns>The normalized text.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="input"/> is null.</exception>
    /// <exception cref="ObjectDisposedException">This instance has been disposed.</exception>
    /// <exception cref="WetextProcessorException">The native processor failed to process the input.</exception>
    public string Normalize(string input)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(input);

        if (input.Length == 0) return string.Empty;
        return InvokeNative(handle => NativeMethods.wetext_normalize(handle, input));
    }

    private string InvokeNative(Func<IntPtr, IntPtr> call)
    {
        bool addedRef = false;
        try
        {
            _handle.DangerousAddRef(ref addedRef);
            var resultPtr = call(_handle.DangerousGetHandle());
            return MarshalNativeString(resultPtr)
                ?? throw new WetextProcessorException(
                    "The native WeTextProcessing processor failed to process the input.");
        }
        finally
        {
            if (addedRef)
            {
                _handle.DangerousRelease();
            }
        }
    }

    /// <summary>
    /// Gets the absolute path to the default model directory
    /// (<see cref="AppContext.BaseDirectory"/> joined with <see cref="DefaultModelRelativeDirectory"/>).
    /// </summary>
    public static string GetDefaultModelDirectory()
        => Path.Combine(AppContext.BaseDirectory, DefaultModelRelativeDirectory);

    private static string GetDefaultModelPath(WetextLanguage language, WetextMode mode, string role)
    {
        var prefix = language switch
        {
            WetextLanguage.Chinese => "zh",
            WetextLanguage.English => "en",
            _ => throw new ArgumentOutOfRangeException(nameof(language), language, "Unsupported WetextLanguage."),
        };
        var direction = mode switch
        {
            WetextMode.TextNormalization => "tn",
            WetextMode.InverseTextNormalization => "itn",
            _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unsupported WetextMode."),
        };
        return Path.Combine(GetDefaultModelDirectory(), $"{prefix}_{direction}_{role}.fst");
    }

    private static string MarshalNativeString(IntPtr nativeStr)
    {
        if (nativeStr == IntPtr.Zero)
        {
            return null;
        }

        try
        {
            return Marshal.PtrToStringUTF8(nativeStr);
        }
        finally
        {
            NativeMethods.wetext_free_string(nativeStr);
        }
    }

    /// <summary>
    /// Releases all resources used by the <see cref="WetextProcessor"/>.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
