using System;
using System.Runtime.InteropServices;

namespace WeTextProcessing.NET.Interop;

internal static partial class NativeMethods
{
    private const string LibraryName = "wetext_processor_c";

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial IntPtr wetext_create_processor(
        string tagger_path,
        string verbalizer_path);

    [LibraryImport(LibraryName)]
    public static partial void wetext_destroy_processor(IntPtr handle);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial IntPtr wetext_tag(
        IntPtr handle,
        string input);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial IntPtr wetext_verbalize(
        IntPtr handle,
        string input);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial IntPtr wetext_normalize(
        IntPtr handle,
        string input);

    [LibraryImport(LibraryName)]
    public static partial void wetext_free_string(IntPtr str);
}