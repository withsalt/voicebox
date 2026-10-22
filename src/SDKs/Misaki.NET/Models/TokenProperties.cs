using System;
using System.Collections.Generic;
using System.Text;

namespace Misaki.NET.Models
{
    /// <summary>
    /// Extended properties for token processing.
    /// </summary>
    public sealed class TokenProperties
    {
        public bool IsHead { get; set; } = true;
        public string? Alias { get; set; }
        public double? Stress { get; set; }
        public string? Currency { get; set; }
        public string NumFlags { get; set; } = string.Empty;
        public bool PreSpace { get; set; }
        public int? Rating { get; set; }
    }
}
