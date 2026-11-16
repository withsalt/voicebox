using Microsoft.Win32.SafeHandles;
using System;
using System.Runtime.InteropServices;
using WeTextProcessing.NET.Interop;

namespace WeTextProcessing.NET.Interop;

internal sealed class WetextProcessorSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public WetextProcessorSafeHandle()
        : base(ownsHandle: true)
    {
    }

    public WetextProcessorSafeHandle(IntPtr existingHandle)
        : base(ownsHandle: true)
    {
        SetHandle(existingHandle);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.wetext_destroy_processor(handle);
        return true;
    }
}