using System;

namespace WeTextProcessing.NET;

/// <summary>
/// The exception that is thrown when an error occurs in the WeTextProcessing native library.
/// </summary>
public class WetextProcessorException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="WetextProcessorException"/> class.
    /// </summary>
    public WetextProcessorException()
        : base("An error occurred in the WeTextProcessing native library.")
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="WetextProcessorException"/> class with a specified error message.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public WetextProcessorException(string message)
        : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="WetextProcessorException"/> class
    /// with a specified error message and a reference to the inner exception that is the cause of this exception.
    /// </summary>
    /// <param name="message">The error message that explains the reason for the exception.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public WetextProcessorException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}